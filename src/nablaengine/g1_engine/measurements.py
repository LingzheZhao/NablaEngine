"""Native solver measurements sampled once per physics substep.

The contact force is evaluated from the final constraint residual already solved
by the forward kernel. Geometry and forces refer to the substep input state.
Wrenches use world axes and moments about the world origin, in N and N m.
The world body's entry contains the terrain reaction. terrain_wrench excludes
self contacts. all_contact_wrench includes both sides of robot self contacts.
"""

DEVICE = r'''
// s: final solver rows; geometry: the same substep's collision geometry (Sim or Hand).
// The solver's efc_jv scratch is dead after factorization, so use the retained
// final residual efc_jar and row_eval rather than reading overwritten scratch.
template <class S, class Geometry>
__device__ void measure_terrain_wrenches(const S& s, const Geometry& geometry, int lane,
                                         float* wrench, int* contact_count,
                                         float* all_wrench = nullptr, int* all_contact_count = nullptr) {
  for (int body = lane; body < NB; body += 32) {
    float force[3] = {}, moment[3] = {}, all_force[3] = {}, all_moment[3] = {};
    int count = 0, all_count = 0;
    for (int c = 0; c < min(s.ncon, MAXCON); ++c) {
      int pair = s.con_pair[c], b1 = c_gbody[PG1(pair)], b2 = c_gbody[PG2(pair)];
      bool terrain = (b1 == 0) != (b2 == 0);
      if (b1 == b2 || (body != b1 && body != b2) || (!terrain && !all_wrench)) continue;
      float row_force[4] = {};
      for (int j = 0; j < s.con_nrow[c]; ++j) {
        int row = s.con_row[c] + j;
        float derivative, curvature, cost;
        row_eval(s, row, s.efc_jar[row], &derivative, &curvature, &cost);
        row_force[j] = -derivative;
      }
      float mu = s.con_dim[c] == 3 ? s.con_mu[c] : 0.f;
      float local[3] = {row_force[0]+row_force[1]+row_force[2]+row_force[3],
                        mu*(row_force[0]-row_force[1]), mu*(row_force[2]-row_force[3])};
      const float* frame = geometry.con_frame[c];
      float sign = body == b2 ? 1.f : -1.f, world[3];
      for (int j = 0; j < 3; ++j) {
        world[j] = sign*(frame[j]*local[0]+frame[3+j]*local[1]+frame[6+j]*local[2]);
        if (terrain) force[j] += world[j];
        if (all_wrench) all_force[j] += world[j];
      }
      const float* position = geometry.con_pos[c];
      float torque[3];
      cross(position, world, torque);
      for (int j = 0; j < 3; ++j) {
        if (terrain) moment[j] += torque[j];
        if (all_wrench) all_moment[j] += torque[j];
      }
      if (terrain) ++count;
      ++all_count;
    }
    for (int j = 0; j < 3; ++j) {
      wrench[6*body+j] = force[j];
      wrench[6*body+3+j] = moment[j];
      if (all_wrench) {
        all_wrench[6*body+j] = all_force[j];
        all_wrench[6*body+3+j] = all_moment[j];
      }
    }
    contact_count[body] = count;
    if (all_contact_count) all_contact_count[body] = all_count;
  }
}
'''
