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
    import torch
    if center.shape != (3,) or axis.shape != (3,) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('expected center/axis[3] and vertices[V,3]')
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError('expected triangulated closed convex hull')
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
    tri = vertices[faces]
    n = torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    norm = torch.linalg.vector_norm(n,dim=1)
    if bool((norm <= tolerance).any()):
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
        aa=torch.dot(e0,e0);ab=torch.dot(e0,e1);bb=torch.dot(e1,e1)
        u=(torch.dot(d,e0)*bb-torch.dot(d,e1)*ab)/(aa*bb-ab*ab)
        v=(torch.dot(d,e1)*aa-torch.dot(d,e0)*ab)/(aa*bb-ab*ab)
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
                de=torch.dot(delta,edge);det=dd*ee-de*de
                if float(det.detach())>tolerance*tolerance:
                    dw=torch.dot(delta,w);ew=torch.dot(edge,w)
                    t=(de*ew-ee*dw)/det;u=(dd*ew-de*dw)/det
                    if 0<float(t.detach())<1 and 0<float(u.detach())<1:
                        add(t,a+edge*u,('edge-interior',i,j))
    if not candidates: raise UnsupportedContact('no external closest feature')
    _,c,q,t,feature=min(candidates,key=lambda item:float(item[0].detach()))
    distance=torch.linalg.vector_norm(q-c)
    if float(distance.detach())<=tolerance:
        raise UnsupportedContact('zero-distance centerline witness')
    normal=(q-c)/distance
    return Contact(distance-radius,(q+c+normal*radius)*.5,normal,t,feature)
