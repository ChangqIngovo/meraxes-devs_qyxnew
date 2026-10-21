#ifndef AG_ANGULAR_FACES_H
#define AG_ANGULAR_FACES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Geodesic triangular bin, with inward unit cone-plane normals.
 * Arrays contain three consecutive xyz vectors.  Vertices have outward
 * counterclockwise winding; solid_angle is in steradians. */
typedef struct {
  double vertices[9];
  double inward_normals[9];
  double solid_angle;
} ag_bin;

/* Eight octants, recursively divided into four great-circle triangles.
 * Levels 0..8 are supported.  Invalid levels return zero. */
size_t ag_bin_count(unsigned level);

/* Fill at least ag_bin_count(level) entries. Return 0 on success, -1 for
 * invalid arguments. No allocation or global mutable state is used. */
int ag_make_bins(unsigned level, ag_bin *bins, size_t capacity);

/* Unsigned source-centred solid angle of a Cartesian square face. The face
 * centre is relative to the source, axis is its normal coordinate (0..2),
 * and halfwidth is half its edge length. Return NAN for invalid input or
 * a face containing the source itself. A coplanar face outside the source
 * has zero solid angle. */
double ag_face_solid_angle(const double center[3], int axis, double halfwidth);

/* Solid angle of that face's radial projection intersected with bin.
 * Clip the planar square by the three cone halfspaces, then integrate its
 * projected spherical polygon. No ray trajectories or angular quadrature
 * are constructed. Shared faces should be cached once by the caller. */
double ag_face_bin_solid_angle(const double center[3], int axis,
                              double halfwidth, const ag_bin *bin);

#ifdef __cplusplus
}
#endif
#endif
