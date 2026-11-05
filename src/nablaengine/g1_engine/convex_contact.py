"""CUDA static convex-hull contact selection and differentiable witness replay.

Slots zero and one expose the minimum and maximum segment parameters of
the near-minimum-distance witness set. A spatially unique witness uses slot zero.
The two-point set retains the endpoints of a parallel face contact interval.
Intersecting centerlines use the minimum-support facet of hull plus segment to
compute signed penetration and recover witnesses at the separating translation.
Callers propagate every negative status to the engine unsupported-state
flag and stop the training batch at the host boundary.

Selection and replay share explicitly rounded primal arithmetic. Six-variable
forward AD replays only the selected witness, including its continuous segment
and edge parameters. Hull geometry and pose are fixed.
"""

DEVICE = r'''
namespace static_convex {
constexpr float geometry_tolerance = 2e-5f;
constexpr float manifold_distance_tolerance = 2e-6f;
enum Status { NoContact = 0, Valid = 1, InvalidGeometry = -3, InvalidSlot = -4 };

// Scalar<0> and Scalar<6> have identical explicitly rounded primal operations.
// This keeps replay stable across compilation units and contraction settings.
template<int D> struct Scalar {
  float value;
  float derivative[D ? D : 1];
  __device__ Scalar(float input = 0.f) : value(input) {
    for (int i = 0; i < D; ++i) derivative[i] = 0.f;
  }
};
template<int D> __device__ Scalar<D> operator+(const Scalar<D>& a, const Scalar<D>& b) {
  Scalar<D> result(__fadd_rn(a.value, b.value));
  for (int i = 0; i < D; ++i) result.derivative[i] = a.derivative[i] + b.derivative[i];
  return result;
}
template<int D> __device__ Scalar<D> operator-(const Scalar<D>& a, const Scalar<D>& b) {
  Scalar<D> result(__fsub_rn(a.value, b.value));
  for (int i = 0; i < D; ++i) result.derivative[i] = a.derivative[i] - b.derivative[i];
  return result;
}
template<int D> __device__ Scalar<D> operator*(const Scalar<D>& a, const Scalar<D>& b) {
  Scalar<D> result(__fmul_rn(a.value, b.value));
  for (int i = 0; i < D; ++i)
    result.derivative[i] = a.derivative[i]*b.value + a.value*b.derivative[i];
  return result;
}
template<int D> __device__ Scalar<D> operator/(const Scalar<D>& a, const Scalar<D>& b) {
  Scalar<D> result(a.value/b.value);
  for (int i = 0; i < D; ++i)
    result.derivative[i] = (a.derivative[i]*b.value-a.value*b.derivative[i])/(b.value*b.value);
  return result;
}
template<int D> __device__ Scalar<D> square_root(const Scalar<D>& a) {
  Scalar<D> result(sqrtf(a.value));
  for (int i = 0; i < D; ++i) result.derivative[i] = a.derivative[i]/(2.f*result.value);
  return result;
}
template<int D> __device__ Scalar<D> clamp_unit(const Scalar<D>& a) {
  if (a.value < 0.f) return Scalar<D>(0.f);
  if (a.value > 1.f) return Scalar<D>(1.f);
  return a;
}
template<int D> struct Vector { Scalar<D> x[3]; };
template<int D> __device__ Vector<D> load_vector(const float* data) {
  Vector<D> result;
  for (int i = 0; i < 3; ++i) result.x[i] = Scalar<D>(data[i]);
  return result;
}
template<int D> __device__ Vector<D> operator+(const Vector<D>& a, const Vector<D>& b) {
  Vector<D> result;
  for (int i = 0; i < 3; ++i) result.x[i] = a.x[i]+b.x[i];
  return result;
}
template<int D> __device__ Vector<D> operator-(const Vector<D>& a, const Vector<D>& b) {
  Vector<D> result;
  for (int i = 0; i < 3; ++i) result.x[i] = a.x[i]-b.x[i];
  return result;
}
template<int D> __device__ Vector<D> operator*(const Vector<D>& a, const Scalar<D>& b) {
  Vector<D> result;
  for (int i = 0; i < 3; ++i) result.x[i] = a.x[i]*b;
  return result;
}
template<int D> __device__ Scalar<D> dot(const Vector<D>& a, const Vector<D>& b) {
  return (a.x[0]*b.x[0]+a.x[1]*b.x[1])+a.x[2]*b.x[2];
}
template<int D> __device__ Vector<D> cross_product(const Vector<D>& a, const Vector<D>& b) {
  Vector<D> result;
  for (int i = 0; i < 3; ++i) {
    int j = (i+1)%3, k = (i+2)%3;
    result.x[i] = a.x[j]*b.x[k]-a.x[k]*b.x[j];
  }
  return result;
}
template<int D> __device__ Vector<D> rotate(const float* quaternion, const Vector<D>& v, bool inverse) {
  float vector_part[3] = {quaternion[1], quaternion[2], quaternion[3]};
  if (inverse) for (int k = 0; k < 3; ++k) vector_part[k] = -vector_part[k];
  Vector<D> q = load_vector<D>(vector_part);
  Vector<D> t = cross_product(q, v)*Scalar<D>(2.f);
  return v+t*Scalar<D>(quaternion[0])+cross_product(q, t);
}
template<int D> __device__ Vector<D> vertex(int hull, int index) {
  return load_vector<D>(c_hull_vertices+3*(c_hull_vertexadr[hull]+index));
}

template<int D> __device__ Vector<D> face_normal(int face) {
  Vector<D> normal = load_vector<D>(c_hull_normals+3*face);
  // Stored float32 components do not have exactly unit norm when widened.
  // Both the signed and separated branches use this same unit normal.
  return normal*(Scalar<D>(1.f)/square_root(dot(normal, normal)));
}

// kind 0: face at a segment endpoint; kind 1: segment endpoint to edge;
// kind 2: hull vertex to segment; kind 3: interior segment to interior edge.
struct Feature { int face, edge, kind, endpoint; };
template<int D> struct Witness {
  Vector<D> centerline, surface;
  Scalar<D> parameter;
};
template<int D> __device__ Witness<D> replay(int hull, const Feature& feature,
                                             const Vector<D>& start, const Vector<D>& delta) {
  const int* triangle = c_hull_faces+3*feature.face;
  Vector<D> a = vertex<D>(hull, triangle[feature.edge]);
  Vector<D> b = vertex<D>(hull, triangle[(feature.edge+1)%3]);
  Vector<D> edge = b-a;
  Scalar<D> t(float(feature.endpoint));
  Vector<D> surface;
  if (feature.kind == 0) {
    Vector<D> center = start+delta*t;
    Vector<D> normal = face_normal<D>(feature.face);
    surface = center-normal*dot(normal, center-a);
  } else if (feature.kind == 1) {
    Vector<D> center = start+delta*t;
    Scalar<D> u = clamp_unit(dot(edge, center-a)/dot(edge, edge));
    surface = a+edge*u;
  } else if (feature.kind == 2) {
    surface = a+edge*Scalar<D>(float(feature.endpoint));
    t = clamp_unit(dot(delta, surface-start)/dot(delta, delta));
  } else {
    Vector<D> normal = cross_product(delta, edge);
    Vector<D> offset = a-start;
    Scalar<D> determinant = dot(normal, normal);
    t = dot(cross_product(offset, edge), normal)/determinant;
    Scalar<D> u = dot(cross_product(offset, delta), normal)/determinant;
    surface = a+edge*u;
  }
  return {start+delta*t, surface, t};
}

// A facet of hull + [-delta/2,+delta/2]. side=1 means the supporting
// plane normal is an exposed hull edge crossed with the capsule direction.
struct PenetrationFacet {
  int face, edge, side, sign, support_vertex, endpoint;
};
template<int D> __device__ void penetration_geometry(int hull, const PenetrationFacet& facet,
                                                       const Vector<D>& start, const Vector<D>& delta,
                                                       Vector<D>& normal, Scalar<D>& depth) {
  if (facet.side) {
    const int* triangle = c_hull_faces+3*facet.face;
    Vector<D> edge = vertex<D>(hull, triangle[(facet.edge+1)%3])-vertex<D>(hull, triangle[facet.edge]);
    normal = cross_product(edge, delta);
    normal = normal*(Scalar<D>(float(facet.sign))/square_root(dot(normal, normal)));
  } else {
    normal = face_normal<D>(facet.face);
  }
  Vector<D> endpoint = start+delta*Scalar<D>(float(facet.endpoint));
  depth = dot(normal, vertex<D>(hull, facet.support_vertex)-endpoint);
}

struct Selection {
  bool penetrating;
  PenetrationFacet penetration_facet;
  Feature feature;
  Witness<0> witness;
  float squared_distance;
  bool found, has_extrema;
  Feature lower_feature, upper_feature;
  Witness<0> lower_witness, upper_witness;
  int status;
};
__device__ void consider(int hull, const Feature& feature, const Vector<0>& start,
                          const Vector<0>& delta, int phase, Selection& selection) {
  Witness<0> candidate = replay(hull, feature, start, delta);
  Vector<0> offset = candidate.surface-candidate.centerline;
  float squared = dot(offset, offset).value;
  if (!isfinite(squared)) { selection.status = InvalidGeometry; return; }
  if (phase == 0) {
    if (!selection.found || squared < selection.squared_distance) {
      selection.feature = feature;
      selection.witness = candidate;
      selection.squared_distance = squared;
      selection.found = true;
    }
    return;
  }
  float best_distance = sqrtf(selection.squared_distance);
  float distance_tolerance = fmaxf(manifold_distance_tolerance,
                                   8.f*1.1920928955078125e-7f*fmaxf(1.f, best_distance));
  if (fabsf(sqrtf(squared)-best_distance) <= distance_tolerance) {
    if (!selection.has_extrema || candidate.parameter.value < selection.lower_witness.parameter.value) {
      selection.lower_feature = feature;
      selection.lower_witness = candidate;
    }
    if (!selection.has_extrema || candidate.parameter.value > selection.upper_witness.parameter.value) {
      selection.upper_feature = feature;
      selection.upper_witness = candidate;
    }
    selection.has_extrema = true;
  }
}
__device__ void candidates(int hull, const Vector<0>& start, const Vector<0>& delta,
                            int phase, Selection& selection) {
  float delta_squared = dot(delta, delta).value;
  for (int i = 0; i < c_hull_facenum[hull]; ++i) {
    int face = c_hull_faceadr[hull]+i;
    const int* triangle = c_hull_faces+3*face;
    Vector<0> a = vertex<0>(hull, triangle[0]);
    Vector<0> b = vertex<0>(hull, triangle[1]);
    Vector<0> c = vertex<0>(hull, triangle[2]);
    Vector<0> normal = face_normal<0>(face);
    Vector<0> e0 = b-a, e1 = c-a;
    Vector<0> triangle_cross = cross_product(e0, e1);
    Scalar<0> triangle_squared = dot(triangle_cross, triangle_cross);
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
      Vector<0> point = start+delta*Scalar<0>(float(endpoint));
      Vector<0> projection = point-normal*dot(normal, point-a);
      Vector<0> offset = projection-a;
      float u = (dot(cross_product(offset, e1), triangle_cross)/triangle_squared).value;
      float v = (dot(cross_product(e0, offset), triangle_cross)/triangle_squared).value;
      if (u >= 0.f && v >= 0.f && __fadd_rn(u, v) <= 1.f)
        consider(hull, {face, 0, 0, endpoint}, start, delta, phase, selection);
    }
    for (int edge_index = 0; edge_index < 3; ++edge_index) {
      for (int endpoint = 0; endpoint < 2; ++endpoint)
        consider(hull, {face, edge_index, 1, endpoint}, start, delta, phase, selection);
      if (delta_squared <= geometry_tolerance*geometry_tolerance) continue;
      for (int endpoint = 0; endpoint < 2; ++endpoint)
        consider(hull, {face, edge_index, 2, endpoint}, start, delta, phase, selection);
      Vector<0> edge_a = vertex<0>(hull, triangle[edge_index]);
      Vector<0> edge_b = vertex<0>(hull, triangle[(edge_index+1)%3]);
      Vector<0> edge = edge_b-edge_a;
      Vector<0> cross = cross_product(delta, edge);
      Scalar<0> determinant = dot(cross, cross);
      if (determinant.value <= 0.f) continue;
      Vector<0> offset = edge_a-start;
      float t = (dot(cross_product(offset, edge), cross)/determinant).value;
      float u = (dot(cross_product(offset, delta), cross)/determinant).value;
      if (t > 0.f && t < 1.f && u > 0.f && u < 1.f)
        consider(hull, {face, edge_index, 3, 0}, start, delta, phase, selection);
    }
  }
}
__device__ void consider_penetration_facet(int hull, int face, int edge_index, bool side,
                                             const Vector<0>& start, const Vector<0>& delta,
                                             bool& found, float& minimum_depth,
                                             PenetrationFacet& selected) {
  Vector<0> normal;
  int sign = 1;
  const int* triangle = c_hull_faces+3*face;
  Vector<0> a = vertex<0>(hull, triangle[edge_index]);
  if (side) {
    Vector<0> edge = vertex<0>(hull, triangle[(edge_index+1)%3])-a;
    normal = cross_product(edge, delta);
    float squared = dot(normal, normal).value;
    if (squared <= 0.f) return;
    normal = normal*(Scalar<0>(1.f)/square_root(dot(normal, normal)));
    if (dot(normal, a).value < 0.f) {
      normal = normal*Scalar<0>(-1.f);
      sign = -1;
    }
  } else {
    normal = face_normal<0>(face);
  }
  float support = -3.402823466e38f;
  float extent = 1.f;
  int support_vertex = 0;
  for (int i = 0; i < c_hull_vertexnum[hull]; ++i) {
    Vector<0> v = vertex<0>(hull, i);
    float projection = dot(normal, v).value;
    if (projection > support) { support = projection; support_vertex = i; }
    for (int k = 0; k < 3; ++k) extent = fmaxf(extent, fabsf(v.x[k].value));
  }
  // Full-vertex support is essential: triangulation diagonals are not exposed
  // polytope edges. The tolerance covers float32 normalization/dot roundoff.
  float support_tolerance = 8.f*1.1920928955078125e-7f*extent;
  if (side && __fsub_rn(support, dot(normal, a).value) > support_tolerance) return;
  int endpoint = dot(normal, delta).value >= 0.f ? 0 : 1;
  PenetrationFacet facet = {face, edge_index, int(side), sign, support_vertex, endpoint};
  Vector<0> reconstructed_normal;
  Scalar<0> depth;
  penetration_geometry(hull, facet, start, delta, reconstructed_normal, depth);
  if (!isfinite(depth.value)) return;
  if (!found || depth.value < minimum_depth) {
    found = true;
    minimum_depth = depth.value;
    selected = facet;
  }
}
__device__ int select_penetration(int hull, const Vector<0>& start,
                                   const Vector<0>& delta, Selection& selection) {
  bool found = false;
  float depth = 0.f;
  for (int i = 0; i < c_hull_facenum[hull]; ++i) {
    int face = c_hull_faceadr[hull]+i;
    consider_penetration_facet(hull, face, 0, false, start, delta,
                               found, depth, selection.penetration_facet);
    for (int edge = 0; edge < 3; ++edge)
      consider_penetration_facet(hull, face, edge, true, start, delta,
                                 found, depth, selection.penetration_facet);
  }
  if (!found) return InvalidGeometry;
  Vector<0> normal;
  Scalar<0> displacement;
  penetration_geometry(hull, selection.penetration_facet, start, delta, normal, displacement);
  Vector<0> translated_start = start+normal*displacement;
  selection.found = false;
  selection.has_extrema = false;
  selection.status = Valid;
  candidates(hull, translated_start, delta, 0, selection);
  if (!selection.found || selection.status != Valid) return InvalidGeometry;
  // The minimum-support plane must move the segment to a real hull witness.
  // Failure to reconstruct it is an explicit algorithm error, never a dropped contact.
  if (selection.squared_distance > geometry_tolerance*geometry_tolerance) return InvalidGeometry;
  candidates(hull, translated_start, delta, 1, selection);
  selection.penetrating = true;
  return selection.status;
}

__device__ int select(int hull, const Vector<0>& start, const Vector<0>& delta, Selection& selection) {
  selection.penetrating = false;
  float lower = 0.f, upper = 1.f;
  for (int i = 0; i < c_hull_facenum[hull]; ++i) {
    int face = c_hull_faceadr[hull]+i;
    Vector<0> normal = face_normal<0>(face);
    Vector<0> a = vertex<0>(hull, c_hull_faces[3*face]);
    float offset = dot(normal, start-a).value;
    float slope = dot(normal, delta).value;
    if (slope == 0.f) {
      if (offset > 0.f) { lower = 1.f; upper = 0.f; break; }
    } else if (slope > 0.f) {
      upper = fminf(upper, (-offset)/slope);
    } else {
      lower = fmaxf(lower, (-offset)/slope);
    }
  }
  if (lower <= upper) return select_penetration(hull, start, delta, selection);
  selection.found = false;
  selection.has_extrema = false;
  selection.status = Valid;
  candidates(hull, start, delta, 0, selection);
  if (selection.status != Valid || !selection.found) return InvalidGeometry;
  // At the declared geometry resolution, use a supporting-facet normal
  // instead of normalizing a nearly zero separation vector.
  if (selection.squared_distance <= geometry_tolerance*geometry_tolerance)
    return select_penetration(hull, start, delta, selection);
  candidates(hull, start, delta, 1, selection);
  return selection.status;
}

__device__ int select_slot(const Selection& selection, int slot, Feature& feature) {
  float best_distance = sqrtf(selection.squared_distance);
  float witness_tolerance = fmaxf(geometry_tolerance,
                                  32.f*1.1920928955078125e-7f*fmaxf(1.f, best_distance));
  Vector<0> dc = selection.upper_witness.centerline-selection.lower_witness.centerline;
  // A point query has one physical closest point on a convex solid. Only
  // distinct centerline positions create a two-point capsule contact set.
  bool distinct = dot(dc, dc).value > witness_tolerance*witness_tolerance;
  if (!distinct) {
    if (slot == 1) return NoContact;
    feature = selection.feature;
  } else {
    feature = slot == 0 ? selection.lower_feature : selection.upper_feature;
  }
  return Valid;
}

template<int D> __device__ void local_inputs(int hull, const float* center, const float* axis,
                                              float half_length, const float* hull_position,
                                              const float* hull_quaternion, Vector<D>& start,
                                              Vector<D>& delta) {
  Vector<D> cp = load_vector<D>(center), a = load_vector<D>(axis);
  if constexpr (D == 6) {
    for (int i = 0; i < 3; ++i) cp.x[i].derivative[i] = 1.f;
    for (int i = 0; i < 3; ++i) {
      int j = (i+1)%3, k = (i+2)%3;
      a.x[j].derivative[3+i] = -axis[k];
      a.x[k].derivative[3+i] = axis[j];
    }
  }
  Vector<D> local_center = rotate(hull_quaternion, cp-load_vector<D>(hull_position), true)
                           -load_vector<D>(c_hull_origin+3*hull);
  Vector<D> local_axis = rotate(hull_quaternion, a, true);
  start = local_center-local_axis*Scalar<D>(half_length);
  delta = local_axis*Scalar<D>(2.f*half_length);
}
template<int D> __device__ void outputs(int hull, const Feature& feature, const Vector<D>& start,
                                         const Vector<D>& delta, float radius, const float* hull_position,
                                         const float* hull_quaternion, const PenetrationFacet* penetration,
                                         Scalar<D>* output) {
  Vector<D> normal, position;
  Scalar<D> signed_distance;
  if (penetration) {
    Vector<D> outward;
    Scalar<D> depth;
    penetration_geometry(hull, *penetration, start, delta, outward, depth);
    Witness<D> witness = replay(hull, feature, start+outward*depth, delta);
    Vector<D> original_centerline = start+delta*witness.parameter;
    normal = outward*Scalar<D>(-1.f);
    // Each endpoint retains its own distance derivative when a parallel contact
    // interval produces two slots. Differentiating only the global minimum
    // depth would incorrectly give both endpoints the same rotational VJP.
    signed_distance = dot(original_centerline-witness.surface, outward)-Scalar<D>(radius);
    position = (witness.surface+original_centerline+normal*Scalar<D>(radius))*Scalar<D>(.5f);
  } else {
    Witness<D> witness = replay(hull, feature, start, delta);
    Vector<D> offset = witness.surface-witness.centerline;
    Scalar<D> distance = square_root(dot(offset, offset));
    normal = offset*(Scalar<D>(1.f)/distance);
    signed_distance = distance-Scalar<D>(radius);
    position = (witness.surface+witness.centerline+normal*Scalar<D>(radius))*Scalar<D>(.5f);
  }
  position = rotate(hull_quaternion, position+load_vector<D>(c_hull_origin+3*hull), false)
             +load_vector<D>(hull_position);
  normal = rotate(hull_quaternion, normal, false);
  Vector<D> basis;
  basis.x[0] = Scalar<D>(0.f);
  bool use_y = normal.x[1].value > -.5f && normal.x[1].value < .5f;
  basis.x[1] = Scalar<D>(use_y ? 1.f : 0.f);
  basis.x[2] = Scalar<D>(use_y ? 0.f : 1.f);
  basis = basis-normal*dot(normal, basis);
  basis = basis*(Scalar<D>(1.f)/square_root(dot(basis, basis)));
  Vector<D> third = cross_product(normal, basis);
  output[0] = signed_distance;
  for (int i = 0; i < 3; ++i) {
    output[1+i] = position.x[i];
    output[4+i] = normal.x[i];
    output[7+i] = basis.x[i];
    output[10+i] = third.x[i];
  }
}
}  // namespace static_convex

// All negative return values must fail the world/batch. Positive-distance valid
// queries are ordinary noncontacts and can be discarded by the pair dispatcher.
__device__ int static_convex_contact(int geom, int slot, const float* center, const float* axis,
                                      float radius, float half_length, const float* hull_position,
                                      const float* hull_quaternion, float* distance, float* position,
                                      float* frame) {
  using namespace static_convex;
  if (slot < 0 || slot > 1) return InvalidSlot;
  int hull = c_ghull[geom];
  if (hull < 0) return InvalidGeometry;
  Vector<0> start, delta;
  local_inputs(hull, center, axis, half_length, hull_position, hull_quaternion, start, delta);
  Selection selected;
  int status = select(hull, start, delta, selected);
  if (status != Valid) return status;
  Feature feature;
  status = select_slot(selected, slot, feature);
  if (status != Valid) return status;
  Scalar<0> output[13];
  outputs(hull, feature, start, delta, radius, hull_position, hull_quaternion,
          selected.penetrating ? &selected.penetration_facet : nullptr, output);
  *distance = output[0].value;
  for (int i = 0; i < 3; ++i) position[i] = output[1+i].value;
  for (int i = 0; i < 9; ++i) frame[i] = output[4+i].value;
  return Valid;
}
__device__ int static_convex_vjp(int geom, int slot, const float* center, const float* axis,
                                  float radius, float half_length, const float* hull_position,
                                  const float* hull_quaternion, const float* cotangent, float* gradient) {
  using namespace static_convex;
  for (int i = 0; i < 6; ++i) gradient[i] = 0.f;
  if (slot < 0 || slot > 1) return InvalidSlot;
  int hull = c_ghull[geom];
  if (hull < 0) return InvalidGeometry;
  Vector<0> start, delta;
  local_inputs(hull, center, axis, half_length, hull_position, hull_quaternion, start, delta);
  Selection selected;
  int status = select(hull, start, delta, selected);
  if (status != Valid) return status;
  Feature feature;
  status = select_slot(selected, slot, feature);
  if (status != Valid) return status;
  Vector<6> differentiable_start, differentiable_delta;
  local_inputs(hull, center, axis, half_length, hull_position, hull_quaternion,
               differentiable_start, differentiable_delta);
  Scalar<6> output[13];
  outputs(hull, feature, differentiable_start, differentiable_delta, radius,
          hull_position, hull_quaternion,
          selected.penetrating ? &selected.penetration_facet : nullptr, output);
  for (int i = 0; i < 6; ++i)
    for (int k = 0; k < 13; ++k) gradient[i] += cotangent[k]*output[k].derivative[i];
  return Valid;
}
'''


def device_source():
    """Source follows packed hull declarations and the engine math intrinsics."""
    return DEVICE
