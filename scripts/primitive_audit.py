"""Small CPU STL surface audit; box equivalence requires closed six-plane surface and volume agreement."""
import struct,json,pathlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
root=pathlib.Path('/tmp/instinct-data/20251116_50cm_kneeClimbStep1')
import yaml
meta=yaml.safe_load((root/'metadata.yaml').read_text())
reports=[]
for terrain in meta['terrains']:
 p=root/terrain['terrain_file']; raw=p.read_bytes(); n=struct.unpack_from('<I',raw,80)[0]
 if len(raw)!=84+50*n: raise ValueError('not binary STL')
 dt=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]); tri=np.frombuffer(raw,dt,count=n,offset=84)['v'].astype(float)
 # STL repeats vertices. Quantization tolerance is stated, not an approximation claim.
 v,inv=np.unique(np.round(tri.reshape(-1,3),7),axis=0,return_inverse=True); f=inv.reshape(-1,3)
 edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]); nc,labels=connected_components(coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(v),len(v))),directed=False)
 out=[]
 for c in range(nc):
  ff=f[labels[f[:,0]]==c]; vv=v[labels==c]; t=v[ff]; cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); norm=np.linalg.norm(cross,axis=1); valid=norm>1e-12; normals=cross[valid]/norm[valid,None]
  # normals canonicalized independent of winding; count parallel face directions
  canonical=normals.copy()
  biggest=np.argmax(abs(canonical),axis=1)
  canonical*=np.where(canonical[np.arange(len(canonical)),biggest]>=0,1,-1)[:,None]
  axes=np.unique(np.round(canonical,6),axis=0)
  ee=np.sort(np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]),axis=1); _,counts=np.unique(ee,axis=0,return_counts=True)
  closed=bool(np.all(counts==2)); box=False; err=None
  if len(axes)==3 and max(abs(np.dot(axes[i],axes[j])) for i in range(3) for j in range(i))<1e-6:
   coords=vv@np.array(axes).T; lo=coords.min(0);hi=coords.max(0)
   # every triangle must lie on one of 6 boundary planes (no internal faces)
   tt=t@np.array(axes).T; plane=np.any(np.all(abs(tt-lo)<1e-6,axis=1)|np.all(abs(tt-hi)<1e-6,axis=1),axis=1)
   surf=norm.sum()/2; expected=2*np.sum((hi-lo)*np.roll(hi-lo,1)); err=float(abs(surf-expected));box=closed and bool(plane.all()) and err<1e-5
  try: hullvol=float(ConvexHull(vv).volume)
  except Exception:hullvol=None
  out.append(dict(vertices=len(vv),faces=len(ff),bounds=[vv.min(0).tolist(),vv.max(0).tolist()],normal_axes=len(axes),closed=closed,nonmanifold_edges=int(sum(counts!=2)),box_surface_equivalent=box,box_surface_area_error=err,convex_hull_volume=hullvol))
 reports.append(dict(terrain_id=terrain['terrain_id'],file=terrain['terrain_file'],faces=n,vertices=len(v),components=out))
print(json.dumps(dict(vertex_rounding_decimals=7,plane_tolerance_m=1e-6,terrains=reports),indent=2))
