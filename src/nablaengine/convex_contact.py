"""Experimental external-centerline contacts against one static convex hull.

CPU/Torch reference primitive only: no engine dispatch or contact manifold claim.
Topology selection uses detached values; continuous witness parameters keep gradients.
"""
from dataclasses import dataclass


class UnsupportedContact(ValueError):
    """The query leaves the declared external-centerline domain."""


@dataclass
class Contact:
    distance: object
    position: object
    normal: object
    segment_parameter: object
    feature: tuple


def capsule_contact(center, axis, half_length, radius, vertices, faces, *, tolerance=1e-9):
    """One closest contact, differentiable in center and axis on a stable feature.

    Inputs are Torch tensors with matching dtype/device; faces is [F,3] integer.
    Axis must be unit length. Hull faces must bound a convex closed polyhedron.
    Radius penetration is supported; centerline intersection/touch is refused.
    No hull/axis normalization or silent geometric repair is performed.
    """
    import math
    import torch
    tensors=(center,axis,vertices,faces)
    if not all(isinstance(value,torch.Tensor) for value in tensors):
        raise ValueError('geometry inputs must be Torch tensors')
    if center.dtype not in (torch.float32,torch.float64):
        raise ValueError('geometry requires float32 or float64')
    if axis.dtype != center.dtype or vertices.dtype != center.dtype:
        raise ValueError('center, axis and vertices must share dtype')
    if any(value.device != center.device for value in tensors):
        raise ValueError('all geometry tensors must share device')
    if center.shape != (3,) or axis.shape != (3,) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('expected center/axis[3] and vertices[V,3]')
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError('expected triangulated closed convex hull')
    if not all(math.isfinite(float(x)) for x in (half_length, radius, tolerance)) or tolerance <= 0:
        raise ValueError('finite scalar dimensions and positive tolerance required')
    if float(half_length) < 0 or float(radius) <= 0 or abs(float(torch.linalg.vector_norm(axis).detach())-1) > 1e-6:
        raise ValueError('positive radius, nonnegative length and unit axis required')
    if not bool(torch.isfinite(center).all() and torch.isfinite(axis).all() and torch.isfinite(vertices).all()):
        raise ValueError('finite geometry required')
    if faces.dtype not in (torch.int32, torch.int64) or int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError('valid integer triangle indices required')
    edges = {}
    for face in faces.detach().cpu().tolist():
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((a, b)))
            edges[edge] = edges.get(edge, 0) + 1
    if any(count != 2 for count in edges.values()):
        raise ValueError('closed two-manifold triangle topology required')
    # Translation is not geometry: perform subtraction and witness construction
    # near the fixed hull, then restore world position only at the output.
    origin=vertices.mean(0).detach()
    vertices=vertices-origin
    center=center-origin
    tri = vertices[faces]
    n = torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    norm = torch.linalg.vector_norm(n,dim=1)
    edge_scale=torch.stack([torch.linalg.vector_norm(tri[:,1]-tri[:,0],dim=1),
                            torch.linalg.vector_norm(tri[:,2]-tri[:,1],dim=1),
                            torch.linalg.vector_norm(tri[:,0]-tri[:,2],dim=1)]).amax(0)
    # Face validity uses relative floating-point resolution, not the query
    # distance tolerance: real hulls can contain thin but resolvable triangles.
    if bool((norm <= torch.finfo(center.dtype).eps*edge_scale.square()).any()):
        raise ValueError('degenerate hull face')
    n = n/norm[:,None]
    centroid=vertices.mean(0)
    sign=torch.where(((tri[:,0]-centroid)*n).sum(1)>=0,1.,-1.)
    n=n*sign[:,None]
    if bool((((vertices[None,:,:]-tri[:,0,None,:])*n[:,None,:]).sum(2)>tolerance).any()):
        raise ValueError('faces do not describe a convex hull')
    # A closed convex body intersects a segment iff all halfspaces admit one t.
    p=center-axis*half_length; delta=axis*(2*half_length)
    lo,hi=0.,1.
    for ni,vi in zip(n.detach(),tri[:,0].detach()):
        offset=float(torch.dot(ni,p.detach()-vi)); slope=float(torch.dot(ni,delta.detach()))
        if abs(slope)<=tolerance:
            if offset>tolerance: lo,hi=1.,0.; break
        elif slope>0: hi=min(hi,(tolerance-offset)/slope)
        else: lo=max(lo,(tolerance-offset)/slope)
    if lo<=hi:
        raise UnsupportedContact('centerline intersects or touches the convex hull')
    candidates=[]
    def add(t,q,feature):
        c=p+delta*t
        candidates.append(((c-q).square().sum(),c,q,t,feature))
    def point_triangle(x,t,idx):
        a,b,c=tri[idx]; normal=n[idx]
        q=x-normal*torch.dot(normal,x-a)
        # Projection in triangle, including coplanar triangle boundaries.
        e0=b-a;e1=c-a;d=q-a
        cross=torch.linalg.cross(e0,e1)
        cross_squared=torch.dot(cross,cross)
        u=torch.dot(torch.linalg.cross(d,e1),cross)/cross_squared
        v=torch.dot(torch.linalg.cross(e0,d),cross)/cross_squared
        if float(u.detach())>=0 and float(v.detach())>=0 and float((u+v).detach())<=1:
            add(t,q,('face',idx))
    zero=center.new_tensor(0.);one=center.new_tensor(1.)
    dd=torch.dot(delta,delta)
    for i in range(len(faces)):
        point_triangle(p,zero,i);point_triangle(p+delta,one,i)
        for j in range(3):
            a=tri[i,j];b=tri[i,(j+1)%3];edge=b-a
            ee=torch.dot(edge,edge);w=p-a
            # Both boundary sets plus the unconstrained interior/interior pair.
            for t in (zero,one):
                u=(torch.dot(edge,p+delta*t-a)/ee).clamp(0,1)
                add(t,a+edge*u,('edge',i,j))
            if float(dd.detach())>tolerance*tolerance:
                for u in (zero,one):
                    t=(torch.dot(delta,a+edge*u-p)/dd).clamp(0,1)
                    add(t,a+edge*u,('vertex',i,j,int(float(u))))
                de=torch.dot(delta,edge);cross_de=torch.linalg.cross(delta,edge);det=torch.dot(cross_de,cross_de)
                if float(det.detach())>0:
                    dw=torch.dot(delta,w);ew=torch.dot(edge,w)
                    t=torch.dot(torch.linalg.cross(-w,edge),cross_de)/det
                    u=torch.dot(torch.linalg.cross(-w,delta),cross_de)/det
                    if 0<float(t.detach())<1 and 0<float(u.detach())<1:
                        add(t,a+edge*u,('edge-interior',i,j))
    if not candidates: raise UnsupportedContact('no external closest feature')
    _,c,q,t,feature=min(candidates,key=lambda item:float(item[0].detach()))
    # Equal distances can describe a continuum of physically different contact
    # points (e.g. a spine parallel to a face), not just duplicate triangles.
    # Do not assign an arbitrary point VJP to that set-valued witness.
    best_distance=float(torch.linalg.vector_norm(q-c).detach())
    witness_tolerance=max(tolerance,32*torch.finfo(center.dtype).eps*max(1.,best_distance))
    for squared,other_c,other_q,_,_ in candidates:
        other_distance=float(torch.sqrt(squared).detach())
        if abs(other_distance-best_distance)<=witness_tolerance:
            if max(float(torch.linalg.vector_norm(other_c-c).detach()),
                   float(torch.linalg.vector_norm(other_q-q).detach()))>witness_tolerance:
                raise UnsupportedContact('nonunique closest witness: contact-point VJP is undefined')
    distance=torch.linalg.vector_norm(q-c)
    if float(distance.detach())<=tolerance:
        raise UnsupportedContact('zero-distance centerline witness')
    normal=(q-c)/distance
    return Contact(distance-radius,(q+c+normal*radius)*.5+origin,normal,t,feature)
