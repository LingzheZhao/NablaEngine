"""Native CUDA first-hit rays with a local triangle BVH per visible mesh.

The model's compiled triangles build a preorder, stack-free BVH once. Every ray
uses current body poses, and retains primitive surfaces and robot self occlusion.
The Python owner keeps packed GPU arrays and the compiled library alive together.
"""
import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .cuda import nvcc


SOURCE = r'''
#include <cuda_runtime.h>
#include <math.h>
__device__ float dot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ float3 add(float3 a,float3 b){return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ float3 sub(float3 a,float3 b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ float3 mul(float3 a,float s){return make_float3(a.x*s,a.y*s,a.z*s);}
__device__ float3 cross(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float3 load3(const float*p){return make_float3(p[0],p[1],p[2]);}
__device__ float3 inverse_rotate(const float*q,float3 v){
  float3 axis=make_float3(-q[1],-q[2],-q[3]);
  float3 t=mul(cross(axis,v),2.f);return add(add(v,mul(t,q[0])),cross(axis,t));
}
__device__ bool bounds_hit(const float*b,float3 origin,float3 direction,float maximum){
  float low=0.f,high=maximum;
  float o[3]={origin.x,origin.y,origin.z},d[3]={direction.x,direction.y,direction.z};
  for(int k=0;k<3;++k){
    if(d[k]==0.f){if(o[k]<b[k]||o[k]>b[k+3])return false;}
    else{float a=(b[k]-o[k])/d[k],c=(b[k+3]-o[k])/d[k];low=fmaxf(low,fminf(a,c));high=fminf(high,fmaxf(a,c));}
  }
  return low<=high;
}
__device__ float triangle_hit(const float*t,float3 origin,float3 direction){
  float3 a=load3(t),e1=sub(load3(t+3),a),e2=sub(load3(t+6),a);
  float3 p=cross(direction,e2);float det=dot(e1,p);
  if(fabsf(det)<=1.1920928955078125e-7f*sqrtf(dot(e1,e1)*dot(e2,e2)))return INFINITY;
  float inv=1.f/det;float3 offset=sub(origin,a);float u=dot(offset,p)*inv;
  if(u<0.f||u>1.f)return INFINITY;
  float3 q=cross(offset,e1);float v=dot(direction,q)*inv;
  if(v<0.f||u+v>1.f)return INFINITY;
  float distance=dot(e2,q)*inv;return distance>=0.f?distance:INFINITY;
}
__device__ float sphere_hit(float3 origin,float3 direction,float radius){
  float projection=dot(origin,direction),disc=projection*projection-dot(origin,origin)+radius*radius;
  if(disc<0.f)return INFINITY;float root=sqrtf(disc);
  float near=-projection-root,far=-projection+root;
  return near>=0.f?near:(far>=0.f?far:INFINITY);
}
__device__ float capsule_hit(float3 o,float3 d,float radius,float half_length){
  float speed=d.x*d.x+d.y*d.y,projection=o.x*d.x+o.y*d.y;
  float disc=projection*projection-speed*(o.x*o.x+o.y*o.y-radius*radius),best=INFINITY;
  if(speed>0.f&&disc>=0.f){
    float root=sqrtf(disc);
    for(int sign=-1;sign<=1;sign+=2){float hit=(-projection+sign*root)/speed;float z=o.z+hit*d.z;
      if(hit>=0.f&&fabsf(z)<=half_length)best=fminf(best,hit);}
  }
  for(int cap=-1;cap<=1;cap+=2){
    float3 shifted=o;shifted.z-=cap*half_length;
    float p=dot(shifted,d),delta=p*p-dot(shifted,shifted)+radius*radius;
    if(delta<0.f)continue;float root=sqrtf(delta);
    for(int sign=-1;sign<=1;sign+=2){float hit=-p+sign*root;
      if(hit>=0.f&&cap*(o.z+hit*d.z)>=half_length)best=fminf(best,hit);}
  }
  return best;
}
__device__ float first_hit(int geom_count,int body_count,const int*geometry,const float*poses,
                             const float*sizes,const float*bounds,const int*nodes,const float*triangles,
                             const float*body_positions,const float*body_quaternions,
                             float3 origin,float3 direction,int&geom_id){
  float best=INFINITY;geom_id=-1;
  for(int g=0;g<geom_count;++g){
    const int*info=geometry+5*g;int body=info[0],kind=info[1];
    float3 o=inverse_rotate(body_quaternions+4*body,sub(origin,load3(body_positions+3*body)));
    float3 d=inverse_rotate(body_quaternions+4*body,direction);
    o=inverse_rotate(poses+7*g+3,sub(o,load3(poses+7*g)));
    d=inverse_rotate(poses+7*g+3,d);
    const float*size=sizes+3*g;float distance=INFINITY;
    if(kind==7){
      int node=info[2],end=info[3];
      while(node<end){
        const int*entry=nodes+3*node;
        if(!bounds_hit(bounds+6*node,o,d,fminf(best,distance))){node=entry[2];continue;}
        if(entry[1]>0){for(int i=0;i<entry[1];++i)distance=fminf(distance,triangle_hit(triangles+9*(entry[0]+i),o,d));node=entry[2];}
        else ++node;
      }
    }else if(kind==2)distance=sphere_hit(o,d,size[0]);
    else if(kind==3)distance=capsule_hit(o,d,size[0],size[1]);
    else if(kind==6){
      float b[6]={-size[0],-size[1],-size[2],size[0],size[1],size[2]};
      float low=-INFINITY,high=INFINITY,ov[3]={o.x,o.y,o.z},dv[3]={d.x,d.y,d.z};bool valid=true;
      for(int k=0;k<3;++k){if(dv[k]==0.f){if(ov[k]<b[k]||ov[k]>b[k+3])valid=false;}
        else{float a=(b[k]-ov[k])/dv[k],c=(b[k+3]-ov[k])/dv[k];low=fmaxf(low,fminf(a,c));high=fminf(high,fmaxf(a,c));}}
      if(valid&&low<=high)distance=low>=0.f?low:(high>=0.f?high:INFINITY);
    }else if(kind==0&&d.z!=0.f){float t=-o.z/d.z;if(t>=0.f)distance=t;}
    if(distance<best){best=distance;geom_id=info[4];}
  }
  return best;
}
__global__ void rays(int batch,int ray_count,int geom_count,int body_count,
                       const int*geometry,const float*poses,const float*sizes,
                       const float*bounds,const int*nodes,const float*triangles,
                       const float*body_positions,const float*body_quaternions,
                       const float*origins,const float*directions,float maximum,float minimum,
                       int max_hops,float epsilon,float*distances,long long*ids){
  int ray=blockIdx.x*blockDim.x+threadIdx.x;if(ray>=batch*ray_count)return;
  int world=ray/ray_count;float3 origin=load3(origins+3*ray),direction=load3(directions+3*ray);
  const float*bp=body_positions+3*body_count*world;const float*bq=body_quaternions+4*body_count*world;
  float traveled=0.f,result=maximum;int id=-1;
  for(int hop=0;hop<=max_hops;++hop){
    int hit_id;float hit=first_hit(geom_count,body_count,geometry,poses,sizes,bounds,nodes,triangles,bp,bq,origin,direction,hit_id);
    if(!isfinite(hit)||hit+traveled>maximum)break;
    if(hit+traveled>=minimum){result=hit+traveled;id=hit_id;break;}
    if(hop==max_hops)break;float advance=hit+epsilon;origin=add(origin,mul(direction,advance));traveled+=advance;
  }
  distances[ray]=result;ids[ray]=id;
}
extern "C" int depth_rays(int batch,int ray_count,int geom_count,int body_count,
                            const int*geometry,const float*poses,const float*sizes,
                            const float*bounds,const int*nodes,const float*triangles,
                            const float*body_positions,const float*body_quaternions,
                            const float*origins,const float*directions,float maximum,float minimum,
                            int max_hops,float epsilon,float*distances,long long*ids,void*stream){
  rays<<<(batch*ray_count+127)/128,128,0,(cudaStream_t)stream>>>(batch,ray_count,geom_count,body_count,
    geometry,poses,sizes,bounds,nodes,triangles,body_positions,body_quaternions,origins,directions,
    maximum,minimum,max_hops,epsilon,distances,ids);return int(cudaGetLastError());
}
'''


def _mesh_bvh(triangles, bounds, nodes, ordered):
    """Preorder tree with subtree escape links, leaves contain at most 8 faces."""
    first_node = len(nodes)
    centroids = triangles.mean(axis=1)
    def visit(indices):
        index = len(nodes)
        subset = triangles[indices]
        lower, upper = subset.min(axis=(0, 1)), subset.max(axis=(0, 1))
        # Outward rounding keeps a triangle on its node boundary visible.
        bounds.append(np.r_[np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)])
        nodes.append([0, 0, 0])
        if len(indices) <= 8:
            nodes[index][:2] = [len(ordered), len(indices)]
            ordered.extend(subset)
        else:
            axis = np.argmax(np.ptp(centroids[indices], axis=0))
            midpoint = len(indices)//2
            order = indices[np.argpartition(centroids[indices, axis], midpoint)]
            visit(order[:midpoint])
            visit(order[midpoint:])
        nodes[index][2] = len(nodes)
    visit(np.arange(len(triangles)))
    return first_node, len(nodes)


class CudaDepthGeometry:
    def __init__(self, geoms, body_count, device):
        geometry, poses, sizes, bounds, nodes, triangles = [], [], [], [], [], []
        meshes = {}
        for geom in geoms:
            root, end = 0, 0
            if geom.triangles is not None:
                # Geoms sharing one mesh tensor reuse its local BVH.
                key = geom.triangles.data_ptr()
                if key not in meshes:
                    meshes[key] = _mesh_bvh(geom.triangles.cpu().numpy(), bounds, nodes, triangles)
                root, end = meshes[key]
            geometry.append([geom.body_id, geom.kind, root, end, geom.geom_id])
            poses.append(np.r_[geom.position.cpu().numpy(), geom.quaternion.cpu().numpy()])
            sizes.append(geom.size.cpu().numpy())
        def tensor(values, dtype):
            return torch.tensor(np.asarray(values), dtype=dtype, device=device).contiguous()
        self.geometry = tensor(geometry, torch.int32)
        self.poses = tensor(poses, torch.float32)
        self.sizes = tensor(sizes, torch.float32)
        self.bounds = tensor(bounds or [np.zeros(6)], torch.float32)
        self.nodes = tensor(nodes or [[0, 0, 0]], torch.int32)
        self.triangles = tensor(triangles or [np.zeros((3, 3))], torch.float32)
        self.body_count = body_count
        self.device = self.geometry.device
        parent = os.environ.get('DK_CACHE_DIR')
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        self.build_directory = Path(tempfile.mkdtemp(prefix='depth-rays-', dir=parent))
        source = self.build_directory/'depth.cu'
        source.write_text(SOURCE)
        capability = torch.cuda.get_device_capability(device)
        command = [nvcc(), '-std=c++17', '-O3', '-shared', '-Xcompiler=-fPIC',
                   f'-arch=sm_{capability[0]}{capability[1]}', str(source),
                   '-o', str(self.build_directory/'depth.so')]
        result = subprocess.run(command, capture_output=True, text=True)
        (self.build_directory/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'Depth CUDA build failed, see {self.build_directory / "build.log"}')
        self.library = ctypes.CDLL(str(self.build_directory/'depth.so'))
        self.launch = self.library.depth_rays
        self.launch.restype = ctypes.c_int
        self.launch.argtypes = ([ctypes.c_int]*4+[ctypes.c_void_p]*10+
                                [ctypes.c_float]*2+[ctypes.c_int, ctypes.c_float]+[ctypes.c_void_p]*3)

    @torch.no_grad()
    def raycast(self, body_positions, body_quaternions, origins, directions,
                *, max_distance, min_distance, max_hops, surface_epsilon):
        inputs = [body_positions, body_quaternions, origins, directions]
        if any(value.dtype != torch.float32 or value.device != self.geometry.device for value in inputs):
            raise ValueError('CUDA depth rays and body poses require float32 tensors on the packed device')
        bp, bq, origin, direction = [value.contiguous() for value in inputs]
        distances = torch.empty(origins.shape[:-1], dtype=torch.float32, device=self.device)
        ids = torch.empty_like(distances, dtype=torch.long)
        arrays = [self.geometry, self.poses, self.sizes, self.bounds, self.nodes, self.triangles,
                  bp, bq, origin, direction]
        stream = torch.cuda.current_stream(self.device)
        error = self.launch(origins.shape[0], origins.shape[1], len(self.geometry), self.body_count,
                            *[ctypes.c_void_p(value.data_ptr()) for value in arrays],
                            max_distance, min_distance, max_hops, surface_epsilon,
                            ctypes.c_void_p(distances.data_ptr()), ctypes.c_void_p(ids.data_ptr()),
                            ctypes.c_void_p(stream.cuda_stream))
        if error:
            raise RuntimeError(f'CUDA depth ray launch failed with error {error}')
        return distances, ids
