// spazsim.cpp -- headless, standalone port of the Ballistica spaz sim math.
//
// NO game code and NO physics library is linked or wrapped. Every equation
// here is a re-implementation of the corresponding math from:
//   * ODE "quickstep" (PGS/SOR LCP solver, finite-step body integrator,
//     contact / amotor / soft-fixed joint rows, island DFS ordering,
//     the shuffled constraint order driven by dRand seed 5432)
//   * scene_v1/node/spaz_node.cc  (balance, stand joint, roller ball, motor,
//     brakes, run gas, jump, dizzy->knockout, neck pose, collide callback)
//   * spaz_physics_tuning.h (bg-limbs tuning, i.e. current game physics)
//
// Single precision everywhere, operand order preserved, build with
// -O2 -ffp-contract=off (no fused multiply-add, no fast-math).
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "precalc_rands.h"

typedef float R;
static const R INF = FLT_MAX;
static const R kStep = 8.0f / 1000.0f;  // kGameStepSeconds
static const int kStepMs = 8;

// ---------------------------------------------------------------- math ----
static inline R dot3(const R* a, const R* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static inline void mul0_331(R* A, const R* B, const R* C) {
  A[0] = dot3(B, C); A[1] = dot3(B + 4, C); A[2] = dot3(B + 8, C);
}
static inline void mul1_331(R* A, const R* B, const R* C) {
  A[0] = B[0] * C[0] + B[4] * C[1] + B[8] * C[2];
  A[1] = B[1] * C[0] + B[5] * C[1] + B[9] * C[2];
  A[2] = B[2] * C[0] + B[6] * C[1] + B[10] * C[2];
}
static inline void muladd0_331(R* A, const R* B, const R* C) {
  A[0] += dot3(B, C); A[1] += dot3(B + 4, C); A[2] += dot3(B + 8, C);
}
static inline void mul0_333(R* A, const R* B, const R* C) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      A[4 * i + j] = B[4 * i] * C[j] + B[4 * i + 1] * C[4 + j] + B[4 * i + 2] * C[8 + j];
}
static inline void mul2_333(R* A, const R* B, const R* C) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) A[4 * i + j] = dot3(B + 4 * i, C + 4 * j);
}
static inline void cross(R* a, const R* b, const R* c) {
  a[0] = b[1] * c[2] - b[2] * c[1];
  a[1] = b[2] * c[0] - b[0] * c[2];
  a[2] = b[0] * c[1] - b[1] * c[0];
}
static inline R vlen(const R* a) { return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
static inline R rsqrt(R x) { return 1.0f / sqrtf(x); }

static void normalize3(R* a) {  // dNormalize3 (active variant)
  R a0 = a[0], a1 = a[1], a2 = a[2], l;
  R aa0 = fabsf(a0), aa1 = fabsf(a1), aa2 = fabsf(a2);
  bool two = false;
  if (aa1 > aa0) {
    if (aa2 > aa1) two = true;
    else { a0 /= aa1; a2 /= aa1; l = rsqrt(a0 * a0 + a2 * a2 + 1);
      a[0] = a0 * l; a[1] = copysignf(l, a1); a[2] = a2 * l; return; }
  } else if (aa2 > aa0) two = true;
  else {
    if (aa0 <= 0) { a[0] = 1; a[1] = 0; a[2] = 0; return; }
    a1 /= aa0; a2 /= aa0; l = rsqrt(a1 * a1 + a2 * a2 + 1);
    a[0] = copysignf(l, a0); a[1] = a1 * l; a[2] = a2 * l; return;
  }
  if (two) { a0 /= aa2; a1 /= aa2; l = rsqrt(a0 * a0 + a1 * a1 + 1);
    a[0] = a0 * l; a[1] = a1 * l; a[2] = copysignf(l, a2); }
}
static void normalize4(R* a) {
  R l = dot3(a, a) + a[3] * a[3];
  if (l > 0) { l = rsqrt(l); a[0] *= l; a[1] *= l; a[2] *= l; a[3] *= l; }
  else { a[0] = 1; a[1] = 0; a[2] = 0; a[3] = 0; }
}
static void planeSpace(const R* n, R* p, R* q) {
  if (fabsf(n[2]) > 0.70710678118654752440f) {
    R a = n[1] * n[1] + n[2] * n[2]; R k = rsqrt(a);
    p[0] = 0; p[1] = -n[2] * k; p[2] = n[1] * k;
    q[0] = a * k; q[1] = -n[0] * p[2]; q[2] = n[0] * p[1];
  } else {
    R a = n[0] * n[0] + n[1] * n[1]; R k = rsqrt(a);
    p[0] = -n[1] * k; p[1] = n[0] * k; p[2] = 0;
    q[0] = -n[2] * p[1]; q[1] = n[2] * p[0]; q[2] = a * k;
  }
}
static void qFromAxisAngle(R* q, R ax, R ay, R az, R angle) {
  R l = ax * ax + ay * ay + az * az;
  if (l > 0.0f) {
    angle *= 0.5f; q[0] = cosf(angle); l = sinf(angle) * rsqrt(l);
    q[1] = ax * l; q[2] = ay * l; q[3] = az * l;
  } else { q[0] = 1; q[1] = 0; q[2] = 0; q[3] = 0; }
}
static void qMul0(R* a, const R* b, const R* c) {
  a[0] = b[0]*c[0] - b[1]*c[1] - b[2]*c[2] - b[3]*c[3];
  a[1] = b[0]*c[1] + b[1]*c[0] + b[2]*c[3] - b[3]*c[2];
  a[2] = b[0]*c[2] + b[2]*c[0] + b[3]*c[1] - b[1]*c[3];
  a[3] = b[0]*c[3] + b[3]*c[0] + b[1]*c[2] - b[2]*c[1];
}
static void qMul1(R* a, const R* b, const R* c) {
  a[0] = b[0]*c[0] + b[1]*c[1] + b[2]*c[2] + b[3]*c[3];
  a[1] = b[0]*c[1] - b[1]*c[0] - b[2]*c[3] + b[3]*c[2];
  a[2] = b[0]*c[2] - b[2]*c[0] - b[3]*c[1] + b[1]*c[3];
  a[3] = b[0]*c[3] - b[3]*c[0] - b[1]*c[2] + b[2]*c[1];
}
static void qMul2(R* a, const R* b, const R* c) {
  a[0] =  b[0]*c[0] + b[1]*c[1] + b[2]*c[2] + b[3]*c[3];
  a[1] = -b[0]*c[1] + b[1]*c[0] - b[2]*c[3] + b[3]*c[2];
  a[2] = -b[0]*c[2] + b[2]*c[0] - b[3]*c[1] + b[1]*c[3];
  a[3] = -b[0]*c[3] + b[3]*c[0] - b[1]*c[2] + b[2]*c[1];
}
static void qToR(R* Rm, const R* q) {
  R qq1 = 2 * q[1] * q[1], qq2 = 2 * q[2] * q[2], qq3 = 2 * q[3] * q[3];
  Rm[0] = 1 - qq2 - qq3; Rm[1] = 2 * (q[1]*q[2] - q[0]*q[3]); Rm[2] = 2 * (q[1]*q[3] + q[0]*q[2]);
  Rm[4] = 2 * (q[1]*q[2] + q[0]*q[3]); Rm[5] = 1 - qq1 - qq3; Rm[6] = 2 * (q[2]*q[3] - q[0]*q[1]);
  Rm[8] = 2 * (q[1]*q[3] - q[0]*q[2]); Rm[9] = 2 * (q[2]*q[3] + q[0]*q[1]); Rm[10] = 1 - qq1 - qq2;
}
static void rToQ(R* q, const R* r) {
  R tr = r[0] + r[5] + r[10], s;
  if (tr >= 0) {
    s = sqrtf(tr + 1); q[0] = 0.5f * s; s = 0.5f * (1.0f / s);
    q[1] = (r[9] - r[6]) * s; q[2] = (r[2] - r[8]) * s; q[3] = (r[4] - r[1]) * s;
  } else {
    int c;
    if (r[5] > r[0]) c = (r[10] > r[5]) ? 2 : 1; else c = (r[10] > r[0]) ? 2 : 0;
    if (c == 0) { s = sqrtf((r[0] - (r[5] + r[10])) + 1); q[1] = 0.5f * s; s = 0.5f * (1.0f / s);
      q[2] = (r[1] + r[4]) * s; q[3] = (r[8] + r[2]) * s; q[0] = (r[9] - r[6]) * s; }
    else if (c == 1) { s = sqrtf((r[5] - (r[10] + r[0])) + 1); q[2] = 0.5f * s; s = 0.5f * (1.0f / s);
      q[3] = (r[6] + r[9]) * s; q[1] = (r[1] + r[4]) * s; q[0] = (r[2] - r[8]) * s; }
    else { s = sqrtf((r[10] - (r[0] + r[5])) + 1); q[3] = 0.5f * s; s = 0.5f * (1.0f / s);
      q[1] = (r[8] + r[2]) * s; q[2] = (r[6] + r[9]) * s; q[0] = (r[4] - r[1]) * s; }
  }
}
static void wToDQ(R* dq, const R* w, const R* q) {
  dq[0] = 0.5f * (-w[0]*q[1] - w[1]*q[2] - w[2]*q[3]);
  dq[1] = 0.5f * ( w[0]*q[0] + w[1]*q[3] - w[2]*q[2]);
  dq[2] = 0.5f * (-w[0]*q[3] + w[1]*q[0] + w[2]*q[1]);
  dq[3] = 0.5f * ( w[0]*q[2] - w[1]*q[1] + w[2]*q[0]);
}
// dMatrix3 3x4 Cholesky inverse, exactly as dInvertPDMatrix(n=3, nskip=4).
static bool invertPD(const R* A, R* Ainv) {
  const int n = 3, ns = 4; R L[12]; memcpy(L, A, sizeof(L)); R recip[3];
  R* aa = L;
  for (int i = 0; i < n; i++) {
    R* bb = L; R* cc = L + i * ns;
    for (int j = 0; j < i; j++) {
      R sum = *cc; R* a = aa; R* b = bb;
      for (int k = j; k; k--) sum -= (*(a++)) * (*(b++));
      *cc = sum * recip[j]; bb += ns; cc++;
    }
    R sum = *cc; R* a = aa;
    for (int k = i; k; k--, a++) sum -= (*a) * (*a);
    if (sum <= 0.0f) return false;
    *cc = sqrtf(sum); recip[i] = 1.0f / (*cc); aa += ns;
  }
  memset(Ainv, 0, sizeof(R) * 12);
  for (int i = 0; i < n; i++) {
    R x[3] = {0, 0, 0}; x[i] = 1; R y[3];
    for (int ii = 0; ii < n; ii++) {
      R sum = 0; for (int k = 0; k < ii; k++) sum += L[ii * ns + k] * y[k];
      y[ii] = (x[ii] - sum) / L[ii * ns + ii];
    }
    for (int ii = n - 1; ii >= 0; ii--) {
      R sum = 0; for (int k = ii + 1; k < n; k++) sum += L[k * ns + ii] * x[k];
      x[ii] = (y[ii] - sum) / L[ii * ns + ii];
    }
    for (int j = 0; j < n; j++) Ainv[j * ns + i] = x[j];
  }
  return true;
}
// dRand (global LCG, seed saved/restored around each constraint shuffle)
static uint64_t g_seed = 5432;
static uint64_t dRand() { g_seed = (1664525ULL * g_seed + 1013904223ULL) & 0xffffffffULL; return g_seed; }
static int dRandInt(int n) { double a = double(n) / 4294967296.0; return (int)(double(dRand()) * a); }

// -------------------------------------------------------- rigid bodies ----
struct Joint;
struct JNode { Joint* joint; struct Body* body; JNode* next; };
struct Body {
  R pos[4], q[4], Rm[12], lvel[4], avel[4], facc[4], tacc[4];
  R mass, invMass, I[12], invI[12];
  bool gravity; int tag; JNode* first;
};
static void bodyInit(Body* b) {
  memset(b, 0, sizeof(Body)); b->mass = 1; b->invMass = 1; b->gravity = true;
  b->I[0] = b->I[5] = b->I[10] = 1; b->invI[0] = b->invI[5] = b->invI[10] = 1;
  b->q[0] = 1; b->Rm[0] = b->Rm[5] = b->Rm[10] = 1;
}
static void setQuat(Body* b, const R* q) { memcpy(b->q, q, 4 * sizeof(R)); qToR(b->Rm, b->q); }
static void setRot(Body* b, const R* Rm) {
  R q[4]; rToQ(q, Rm); normalize4(q); memcpy(b->q, q, 4 * sizeof(R)); qToR(b->Rm, b->q);
}
static void setMass(Body* b, R mass, R i0, R i1, R i2) {
  memset(b->I, 0, sizeof(b->I)); b->mass = mass; b->I[0] = i0; b->I[5] = i1; b->I[10] = i2;
  if (!invertPD(b->I, b->invI)) { memset(b->invI, 0, sizeof(b->invI)); b->invI[0] = b->invI[5] = b->invI[10] = 1; }
  b->invMass = 1.0f / b->mass;
}
static void massSphere(Body* b, R density, R radius) {  // dMassSetSphere (double M_PI inside)
  double t = (double)(4.0f / 3.0f) * M_PI * radius * radius * radius * density;
  R total = (R)t; R II = 0.4f * total * radius * radius; setMass(b, total, II, II, II);
}
static void massBox(Body* b, R density, R lx, R ly, R lz) {
  R total = lx * ly * lz * density;
  setMass(b, total, total / 12.0f * (ly * ly + lz * lz), total / 12.0f * (lx * lx + lz * lz),
          total / 12.0f * (lx * lx + ly * ly));
}

// ------------------------------------------------------------- joints ----
enum { J_FIXEDEF, J_AMOTOR, J_CONTACT };
struct Contact {
  R pos[4], normal[4], depth; R mu, bounce, bounce_vel, soft_erp, soft_cfm;
};
struct Joint {
  int type; JNode node[2]; bool reverse; int tag;
  // fixed-EF
  R qrel[4], a1[4], a2[4], ls, ld, as, ad;
  // amotor
  R axis[3][4]; int rel[3]; R vel[3], fmax[3], ncfm[3]; int num;
  // contact
  Contact c; int the_m;
};
struct Info1 { int m, nub; };
struct Info2 { R *J1l, *J1a, *J2l, *J2a, *c, *cfm, *lo, *hi; int* findex; int rowskip; R fps, erp; };

static void attach(Joint* j, Body* b1, Body* b2) {
  if (b1 == nullptr) { b1 = b2; b2 = nullptr; j->reverse = true; } else j->reverse = false;
  j->node[0].joint = j; j->node[1].joint = j; j->node[0].body = b1; j->node[1].body = b2;
  if (b1) { j->node[1].next = b1->first; b1->first = &j->node[1]; } else j->node[1].next = nullptr;
  if (b2) { j->node[0].next = b2->first; b2->first = &j->node[0]; } else j->node[0].next = nullptr;
}
static void detachContact(Joint* j) {  // contacts are always the head of their lists; undo in reverse order
  Body* b1 = j->node[0].body; Body* b2 = j->node[1].body;
  if (b2) b2->first = j->node[0].next;
  if (b1) b1->first = j->node[1].next;
}
static void calcERPCFM(R stiffness, R damping, R step, R* erp, R* cfm) {
  if (stiffness <= 0.0f && damping <= 0.0f) { *erp = 0.0f; *cfm = 9999999999.0f; }
  else { *erp = (step * stiffness) / ((step * stiffness) + damping); *cfm = 1.0f / ((step * stiffness) + damping); }
}
static const R kGlobalCFM = 1e-5f;
static const R kMinDepth = 0.001f;  // dWorldSetContactSurfaceLayer

static void fixedInfo1(Joint* j, Info1* i) {
  i->m = 0; i->nub = 0;
  if (j->ls > 0.0f || j->ld > 0.0f) { i->m += 3; i->nub += 3; }
  if (j->as > 0.0f || j->ad > 0.0f) { i->m += 3; i->nub += 3; }
}
static void fixedInfo2(Joint* j, Info2* info) {
  R orig_erp = info->erp; int offs = 0; int s = info->rowskip;
  Body* b0 = j->node[0].body; Body* b1 = j->node[1].body;
  if (j->ls > 0.0f || j->ld > 0.0f) {
    R erp, cfm; calcERPCFM(j->ls, j->ld, kStep, &erp, &cfm); info->erp = erp;
    info->J1l[0] = 1; info->J1l[s + 1] = 1; info->J1l[2 * s + 2] = 1;
    R a1[4], a2[4]; mul0_331(a1, b0->Rm, j->a1);
    info->J1a[1] = +a1[2]; info->J1a[2] = -a1[1]; info->J1a[s + 0] = -a1[2];
    info->J1a[s + 2] = +a1[0]; info->J1a[2 * s + 0] = +a1[1]; info->J1a[2 * s + 1] = -a1[0];
    info->J2l[0] = -1; info->J2l[s + 1] = -1; info->J2l[2 * s + 2] = -1;
    mul0_331(a2, b1->Rm, j->a2);
    info->J2a[1] = -a2[2]; info->J2a[2] = +a2[1]; info->J2a[s + 0] = +a2[2];
    info->J2a[s + 2] = -a2[0]; info->J2a[2 * s + 0] = -a2[1]; info->J2a[2 * s + 1] = +a2[0];
    R k = info->fps * info->erp;
    for (int q = 0; q < 3; q++) info->c[q] = k * (a2[q] + b1->pos[q] - a1[q] - b0->pos[q]);
    info->cfm[0] = cfm; info->cfm[1] = cfm; info->cfm[2] = cfm; offs += 3;
  }
  if (j->as > 0.0f || j->ad > 0.0f) {
    R erp, cfm; calcERPCFM(j->as, j->ad, kStep, &erp, &cfm); info->erp = erp;
    int st = offs * s;
    info->J1a[st] = 1; info->J1a[st + s + 1] = 1; info->J1a[st + s * 2 + 2] = 1;
    info->J2a[st] = -1; info->J2a[st + s + 1] = -1; info->J2a[st + s * 2 + 2] = -1;
    R qerr[4], e[4], qq[4];
    qMul1(qq, b0->q, b1->q); qMul2(qerr, qq, j->qrel);
    if (qerr[0] < 0) { qerr[1] = -qerr[1]; qerr[2] = -qerr[2]; qerr[3] = -qerr[3]; }
    mul0_331(e, b0->Rm, qerr + 1);
    R k = info->fps * info->erp;
    info->c[offs] = 2 * k * e[0]; info->c[offs + 1] = 2 * k * e[1]; info->c[offs + 2] = 2 * k * e[2];
    info->cfm[offs] = cfm; info->cfm[offs + 1] = cfm; info->cfm[offs + 2] = cfm;
  }
  info->erp = orig_erp;
}
static void amotorAxes(Joint* j, R ax[3][4]) {
  for (int i = 0; i < j->num; i++) {
    if (j->rel[i] == 1) mul0_331(ax[i], j->node[0].body->Rm, j->axis[i]);
    else if (j->rel[i] == 2) { if (j->node[1].body) mul0_331(ax[i], j->node[1].body->Rm, j->axis[i]); }
    else { ax[i][0] = j->axis[i][0]; ax[i][1] = j->axis[i][1]; ax[i][2] = j->axis[i][2]; }
  }
}
static void amotorInfo1(Joint* j, Info1* info) {
  info->m = 0; info->nub = 0;
  for (int i = 0; i < j->num; i++) if (j->fmax[i] > 0) info->m++;  // no joint limits in user mode
}
static void amotorInfo2(Joint* j, Info2* info) {
  R ax[3][4]; amotorAxes(j, ax); int row = 0;
  for (int i = 0; i < j->num; i++) {
    if (!(j->fmax[i] > 0)) continue;
    int srow = row * info->rowskip;
    info->J1a[srow + 0] = ax[i][0]; info->J1a[srow + 1] = ax[i][1]; info->J1a[srow + 2] = ax[i][2];
    if (j->node[1].body) { info->J2a[srow + 0] = -ax[i][0]; info->J2a[srow + 1] = -ax[i][1]; info->J2a[srow + 2] = -ax[i][2]; }
    info->cfm[row] = j->ncfm[i]; info->c[row] = j->vel[i];
    info->lo[row] = -j->fmax[i]; info->hi[row] = j->fmax[i]; row++;
  }
}
static void contactInfo1(Joint* j, Info1* info) {
  int m = 1; if (j->c.mu < 0) j->c.mu = 0; if (j->c.mu > 0) m += 2;
  j->the_m = m; info->m = m; info->nub = 0;
}
static void contactInfo2(Joint* j, Info2* info) {
  int s = info->rowskip, s2 = 2 * s; R normal[4];
  Body* b0 = j->node[0].body; Body* b1 = j->node[1].body;
  for (int i = 0; i < 3; i++) normal[i] = j->reverse ? -j->c.normal[i] : j->c.normal[i];
  normal[3] = 0; R c1[4], c2[4];
  for (int i = 0; i < 3; i++) c1[i] = j->c.pos[i] - b0->pos[i];
  info->J1l[0] = normal[0]; info->J1l[1] = normal[1]; info->J1l[2] = normal[2];
  cross(info->J1a, c1, normal);
  if (b1) {
    for (int i = 0; i < 3; i++) c2[i] = j->c.pos[i] - b1->pos[i];
    info->J2l[0] = -normal[0]; info->J2l[1] = -normal[1]; info->J2l[2] = -normal[2];
    R t[3]; cross(t, c2, normal); info->J2a[0] = -t[0]; info->J2a[1] = -t[1]; info->J2a[2] = -t[2];
  }
  R erp = j->c.soft_erp; R k = info->fps * erp;
  R depth = j->c.depth - kMinDepth; if (depth < 0) depth = 0;
  R pushout = k * depth + 0.0f; info->c[0] = pushout;  // max_vel = inf
  info->cfm[0] = j->c.soft_cfm;
  R outgoing = dot3(info->J1l, b0->lvel) + dot3(info->J1a, b0->avel);
  if (b1) outgoing += dot3(info->J2l, b1->lvel) + dot3(info->J2a, b1->avel);
  if (j->c.bounce_vel >= 0 && (-outgoing) > j->c.bounce_vel) {
    R newc = -j->c.bounce * outgoing + 0.0f; if (newc > info->c[0]) info->c[0] = newc;
  }
  info->lo[0] = 0; info->hi[0] = INF;
  R t1[4], t2[4];
  if (j->the_m >= 2) {
    planeSpace(normal, t1, t2);
    info->J1l[s + 0] = t1[0]; info->J1l[s + 1] = t1[1]; info->J1l[s + 2] = t1[2];
    cross(info->J1a + s, c1, t1);
    if (b1) { info->J2l[s + 0] = -t1[0]; info->J2l[s + 1] = -t1[1]; info->J2l[s + 2] = -t1[2];
      R t[3]; cross(t, c2, t1); info->J2a[s + 0] = -t[0]; info->J2a[s + 1] = -t[1]; info->J2a[s + 2] = -t[2]; }
    info->lo[1] = -j->c.mu; info->hi[1] = j->c.mu; info->findex[1] = 0;
  }
  if (j->the_m >= 3) {
    info->J1l[s2 + 0] = t2[0]; info->J1l[s2 + 1] = t2[1]; info->J1l[s2 + 2] = t2[2];
    cross(info->J1a + s2, c1, t2);
    if (b1) { info->J2l[s2 + 0] = -t2[0]; info->J2l[s2 + 1] = -t2[1]; info->J2l[s2 + 2] = -t2[2];
      R t[3]; cross(t, c2, t2); info->J2a[s2 + 0] = -t[0]; info->J2a[s2 + 1] = -t[1]; info->J2a[s2 + 2] = -t[2]; }
    info->lo[2] = -j->c.mu; info->hi[2] = j->c.mu; info->findex[2] = 0;
  }
}
static void getInfo1(Joint* j, Info1* i) {
  if (j->type == J_FIXEDEF) fixedInfo1(j, i); else if (j->type == J_AMOTOR) amotorInfo1(j, i); else contactInfo1(j, i);
}
static void getInfo2(Joint* j, Info2* i) {
  if (j->type == J_FIXEDEF) fixedInfo2(j, i); else if (j->type == J_AMOTOR) amotorInfo2(j, i); else contactInfo2(j, i);
}

// ---------------------------------------------------- quickstep solver ----
struct World {
  R gravity[3] = {0, -20, 0}; int iters = 10; R sor_w = 1.3f;
};
static void stepBody(Body* b, R h) {
  for (int j = 0; j < 3; j++) b->pos[j] += h * b->lvel[j];
  R dq[4]; wToDQ(dq, b->avel, b->q);
  for (int j = 0; j < 4; j++) b->q[j] += h * dq[j];
  normalize4(b->q); qToR(b->Rm, b->q);
}
struct IdxErr { int index; };

static void sorLCP(int m, int nb, R* J, int* jb, Body** body, R* invI, R* fc, R* b, R* lo, R* hi,
                   R* cfm, int* findex, const World& w) {
  (void)nb; std::vector<R> lambda(m, 0.0f), hicopy(hi, hi + m), iMJ((size_t)m * 12), Ad(m);
  for (int i = 0; i < m; i++) lambda[i] *= 0.9f;  // WARM_STARTING defined: lambda starts at 0
  // compute invM*J^T
  { R* ip = iMJ.data(); R* jp = J;
    for (int i = 0; i < m; i++) {
      int b1 = jb[i * 2], b2 = jb[i * 2 + 1]; R k = body[b1]->invMass;
      for (int j = 0; j < 3; j++) ip[j] = k * jp[j];
      mul0_331(ip + 3, invI + 12 * b1, jp + 3);
      if (b2 >= 0) { k = body[b2]->invMass; for (int j = 0; j < 3; j++) ip[j + 6] = k * jp[j + 6];
        mul0_331(ip + 9, invI + 12 * b2, jp + 9); }
      jp += 12; ip += 12; } }
  memset(fc, 0, sizeof(R) * nb * 6);
  { R* ip = iMJ.data(); R* jp = J;
    for (int i = 0; i < m; i++) {
      R sum = 0; for (int j = 0; j < 6; j++) sum += ip[j] * jp[j];
      if (jb[i * 2 + 1] >= 0) for (int j = 6; j < 12; j++) sum += ip[j] * jp[j];
      ip += 12; jp += 12; Ad[i] = w.sor_w / (sum + cfm[i]); } }
  { R* jp = J; for (int i = 0; i < m; i++) { for (int j = 0; j < 12; j++) { jp[0] *= Ad[i]; jp++; } b[i] *= Ad[i]; } }
  for (int i = 0; i < m; i++) Ad[i] *= cfm[i];
  std::vector<IdxErr> order(m); { int j = 0;
    for (int i = 0; i < m; i++) if (findex[i] < 0) order[j++].index = i;
    for (int i = 0; i < m; i++) if (findex[i] >= 0) order[j++].index = i; }
  for (int it = 0; it < w.iters; it++) {
    uint64_t oldSeed = g_seed;  // RANDOMLY_REORDER_CONSTRAINTS is #defined (to 0): shuffle is active
    if ((it & 7) == 0) for (int i = 1; i < m; ++i) { IdxErr tmp = order[i]; int sw = dRandInt(i + 1); order[i] = order[sw]; order[sw] = tmp; }
    g_seed = oldSeed;
    for (int i = 0; i < m; i++) {
      int index = order[i].index; R* jp = J + index * 12; R* ip = iMJ.data() + index * 12;
      if (findex[index] >= 0) { hi[index] = fabsf(hicopy[index] * lambda[findex[index]]); lo[index] = -hi[index]; }
      int b1 = jb[index * 2], b2 = jb[index * 2 + 1];
      R delta = b[index] - lambda[index] * Ad[index]; R* fp = fc + 6 * b1;
      delta -= fp[0] * jp[0] + fp[1] * jp[1] + fp[2] * jp[2] + fp[3] * jp[3] + fp[4] * jp[4] + fp[5] * jp[5];
      if (b2 >= 0) { fp = fc + 6 * b2;
        delta -= fp[0] * jp[6] + fp[1] * jp[7] + fp[2] * jp[8] + fp[3] * jp[9] + fp[4] * jp[10] + fp[5] * jp[11]; }
      R nl = lambda[index] + delta;
      if (nl < lo[index]) { delta = lo[index] - lambda[index]; lambda[index] = lo[index]; }
      else if (nl > hi[index]) { delta = hi[index] - lambda[index]; lambda[index] = hi[index]; }
      else lambda[index] = nl;
      fp = fc + 6 * b1;
      for (int q = 0; q < 6; q++) fp[q] += delta * ip[q];
      if (b2 >= 0) { fp = fc + 6 * b2; for (int q = 0; q < 6; q++) fp[q] += delta * ip[6 + q]; }
    }
  }
}

static void quickStep(const World& w, Body** body, int nb, Joint** joint0, int nj) {
  R stepsize1 = 1.0f / kStep;
  for (int i = 0; i < nb; i++) body[i]->tag = i;
  std::vector<Joint*> joint(joint0, joint0 + nj);
  std::vector<R> invI((size_t)nb * 12);
  for (int i = 0; i < nb; i++) {
    R tmp[12], I[12];
    mul2_333(tmp, body[i]->I, body[i]->Rm); mul0_333(I, body[i]->Rm, tmp);
    mul2_333(tmp, body[i]->invI, body[i]->Rm); mul0_333(&invI[i * 12], body[i]->Rm, tmp);
  }
  for (int i = 0; i < nb; i++) if (body[i]->gravity) {
    body[i]->facc[0] += body[i]->mass * w.gravity[0]; body[i]->facc[1] += body[i]->mass * w.gravity[1];
    body[i]->facc[2] += body[i]->mass * w.gravity[2]; }
  std::vector<Info1> info(nj); int n2 = 0;
  for (int j = 0; j < nj; j++) { getInfo1(joint[j], &info[n2]); if (info[n2].m > 0) { joint[n2] = joint[j]; n2++; } }
  nj = n2; std::vector<int> ofs(nj); int m = 0;
  for (int i = 0; i < nj; i++) { ofs[i] = m; m += info[i].m; }
  if (m > 0) {
    std::vector<R> J((size_t)m * 12, 0.0f), c(m, 0.0f), cfm(m, kGlobalCFM), lo(m, -INF), hi(m, INF);
    std::vector<int> findex(m, -1), jb((size_t)m * 2);
    Info2 ji; ji.rowskip = 12; ji.fps = stepsize1; ji.erp = 0.2f;
    for (int i = 0; i < nj; i++) {
      ji.J1l = &J[ofs[i] * 12]; ji.J1a = ji.J1l + 3; ji.J2l = ji.J1l + 6; ji.J2a = ji.J1l + 9;
      ji.c = &c[ofs[i]]; ji.cfm = &cfm[ofs[i]]; ji.lo = &lo[ofs[i]]; ji.hi = &hi[ofs[i]]; ji.findex = &findex[ofs[i]];
      getInfo2(joint[i], &ji);
      for (int j = 0; j < info[i].m; j++) if (findex[ofs[i] + j] >= 0) findex[ofs[i] + j] += ofs[i];
    }
    { int* jp = jb.data();
      for (int i = 0; i < nj; i++) {
        int b1 = joint[i]->node[0].body ? joint[i]->node[0].body->tag : -1;
        int b2 = joint[i]->node[1].body ? joint[i]->node[1].body->tag : -1;
        for (int j = 0; j < info[i].m; j++) { jp[0] = b1; jp[1] = b2; jp += 2; } } }
    std::vector<R> tmp1((size_t)nb * 6), rhs(m);
    for (int i = 0; i < nb; i++) {
      R inv = body[i]->invMass;
      for (int j = 0; j < 3; j++) tmp1[i * 6 + j] = body[i]->facc[j] * inv + body[i]->lvel[j] * stepsize1;
      mul0_331(&tmp1[i * 6 + 3], &invI[i * 12], body[i]->tacc);
      for (int j = 0; j < 3; j++) tmp1[i * 6 + 3 + j] += body[i]->avel[j] * stepsize1;
    }
    { R* jp = J.data();
      for (int i = 0; i < m; i++) {
        int b1 = jb[i * 2], b2 = jb[i * 2 + 1]; R sum = 0; R* in = &tmp1[b1 * 6];
        for (int j = 0; j < 6; j++) sum += jp[j] * in[j];
        jp += 6;
        if (b2 >= 0) { in = &tmp1[b2 * 6]; for (int j = 0; j < 6; j++) sum += jp[j] * in[j]; }
        jp += 6; rhs[i] = sum; } }
    for (int i = 0; i < m; i++) rhs[i] = c[i] * stepsize1 - rhs[i];
    for (int i = 0; i < m; i++) cfm[i] *= stepsize1;
    std::vector<R> cforce((size_t)nb * 6);
    sorLCP(m, nb, J.data(), jb.data(), body, invI.data(), cforce.data(), rhs.data(), lo.data(), hi.data(),
           cfm.data(), findex.data(), w);
    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < 3; j++) body[i]->lvel[j] += kStep * cforce[i * 6 + j];
      for (int j = 0; j < 3; j++) body[i]->avel[j] += kStep * cforce[i * 6 + 3 + j];
    }
  }
  for (int i = 0; i < nb; i++) {
    R inv = body[i]->invMass;
    for (int j = 0; j < 3; j++) body[i]->lvel[j] += kStep * inv * body[i]->facc[j];
    for (int j = 0; j < 3; j++) body[i]->tacc[j] *= kStep;
    muladd0_331(body[i]->avel, &invI[i * 12], body[i]->tacc);
  }
  for (int i = 0; i < nb; i++) stepBody(body[i], kStep);
  for (int i = 0; i < nb; i++) { memset(body[i]->facc, 0, 3 * sizeof(R)); memset(body[i]->tacc, 0, 3 * sizeof(R)); }
}

// dxProcessIslands for a world whose every body is enabled.
static void processIslands(const World& w, Body** allBodiesFirstToLast, int nbodies, std::vector<Joint*>& allJoints) {
  for (int i = 0; i < nbodies; i++) allBodiesFirstToLast[i]->tag = 0;
  for (Joint* j : allJoints) j->tag = 0;
  std::vector<Body*> body(nbodies); std::vector<Joint*> joint(allJoints.size()); std::vector<Body*> stack(nbodies + 1);
  for (int bi = 0; bi < nbodies; bi++) {
    Body* bb = allBodiesFirstToLast[bi]; if (bb->tag) continue; bb->tag = 1;
    int stacksize = 0, bcount = 1, jcount = 0; Body* b = bb; body[0] = bb; bool quick = true;
    while (quick || stacksize > 0) {
      if (!quick) { b = stack[--stacksize]; body[bcount++] = b; }
      quick = false;
      for (JNode* n = b->first; n; n = n->next) {
        if (!n->joint->tag) { n->joint->tag = 1; joint[jcount++] = n->joint;
          if (n->body && !n->body->tag) { n->body->tag = 1; stack[stacksize++] = n->body; } }
      }
    }
    quickStep(w, body.data(), bcount, joint.data(), jcount);
    for (int i = 0; i < bcount; i++) body[i]->tag = 1;
    for (int i = 0; i < jcount; i++) joint[i]->tag = 1;
  }
}

// ----------------------------------------------------------- tuning ----
struct Tuning {  // kSpazPhysicsTuningBgLimbs
  R head_density = 1.0f, torso_density = 3.65f, pelvis_density = 6.25f, pelvis_mass_height = 0.264f, roller_density = 0.125f;
  R roller_joint_stiffness = 1500.0f, roller_joint_damping = 6.0f, jump_push_distance = 0.3f; int jump_steps = 7;
  R jump_roller_stiffness_scale = 0.4f, jump_roller_damping_scale = 0.2f;
  R pelvis_linear_stiffness = 400.0f, pelvis_linear_damping = 40.0f, pelvis_angular_stiffness = 1.5f, pelvis_angular_damping = 0.06f;
  R neck_linear_stiffness = 500.0f, neck_linear_damping = 1.0f, neck_angular_stiffness = 13.0f, neck_angular_damping = 0.8f;
  R neck_ko_linear_stiffness = 400.0f, neck_ko_linear_damping = 1.0f, neck_ko_angular_stiffness = 5.0f, neck_ko_angular_damping = 0.3f;
  R stand_angular_stiffness = 270.0f, stand_angular_damping = 3.5f, stand_balance_scale = 0.6f, stand_balance_scale_holding = 0.9f, lean_amount = 0.4f;
  R roller_motor_max_force_low_gear = 15.0f, roller_motor_max_force_high_gear = 15.0f, roller_motor_max_force_vertical = 500.0f;
  R walk_speed = 7.68f, run_speed_add = 15.0f, gear_up_speed = 7.0f, speed_smoothing_up = 0.985f, speed_smoothing_down = 0.94f;
  R uphill_slow = 0.2f, downhill_boost = 0.1f, brake_max_force = 10.0f, brake_idle_scale = 0.4f;
  R core_contact_stiffness = 6000.0f, core_contact_damping = 0.01f, roller_floor_contact_stiffness = 7000.0f,
    roller_floor_contact_damping = 10.0f, roller_side_contact_stiffness = 2000.0f, roller_side_contact_damping = 0.005f,
    roller_wall_kick_force = 100.0f, body_friction_scale = 0.1f;
  R down_ball_size = 0.5f, down_ball_bottom_lift = 0.108f, down_ball_contact_stiffness = 1600.0f, down_ball_contact_damping = 8.0f;
  bool down_ball_brakes = true; int wake_slide_steps = 25;
};
// rig constants (bacommon/characterrig.py SPAZ_RIG)
static const R kHeadRadius = 0.23f, kRollerRadius = 0.3f, kPelvisSize[3] = {0.35f, 0.16f, 0.1f};
static const R kStandHead = 2.25f, kStandTorso = 1.8f, kStandPelvis = 1.66f, kStandRoller = 1.6f;
static const R kPelvisJointDrop = 0.05f, kPelvisJointForward = 0.05f;

// ------------------------------------------------------------ the map ----
struct Box { R mn[3], mx[3]; };

// ----------------------------------------------------------- the sim ----
enum { B_HEAD, B_TORSO, B_PELVIS, B_ROLLER, B_STAND };
struct Sim {
  Tuning t; World world; Body bodies[5]; Joint joints[6]; std::vector<Joint> contacts;
  Joint *neck, *pelvisJ, *rollerJ, *standJ, *motorRoller, *motorBrakes;
  std::vector<Box> map; std::vector<Joint*> alljoints;
  R torso_radius = 0.15f, roller_geom_radius = 0.3f, base_roller_anchor = 0;
  // spaz node state
  int8_t lr = 0, ud = 0; R run = 0; bool jump_pressed = false;
  R ud_smooth = 0, lr_smooth = 0, ud_norm = 0, lr_norm = 0, ud_diff_smooth = 0, lr_diff_smooth = 0, ud_diff_smoother = 0, lr_diff_smoother = 0;
  R run_gas = 0, a_vel_y_smoothed = 0, a_vel_y_smoothed_more = 0, speed_smoothed = 0, accel[3] = {0, 0, 0}, prev_vel[3] = {0, 0, 0};
  R last_orient_x = 0, last_orient_z = 0, roll_amt = 0, ball_size = 1.0f, idle_sway = 0.05f;
  uint8_t balance = 0, dizzy = 0, knockout = 0, wake_slide = 0, jump = 0; int footing = 0, fly_time = 0;
  bool running = false, head_back = false; int64_t stepnum = 0; int behavior_version = 2;
  int steps_age = 0;

  Sim() { contacts.reserve(512); }

  // RigidBody::SetDimensions equivalents (mass half; geom radius kept on the side)
  void setSphereMass(Body* b, R m1, R density_mult) { massSphere(b, 5.0f * density_mult, m1); }
  void setBoxMass(Body* b, R m1, R m2, R m3, R density_mult) { massBox(b, 5.0f * density_mult, m1, m2, m3); }

  void build() {
    for (int i = 0; i < 5; i++) bodyInit(&bodies[i]);
    bodies[B_STAND].gravity = false;
    // SetDimensions(...) of the constructor + UpdateBodiesForStyle
    setSphereMass(&bodies[B_HEAD], 0.28f, t.head_density);
    setSphereMass(&bodies[B_TORSO], 0.2f, t.torso_density);
    setBoxMass(&bodies[B_PELVIS], 0.25f, t.pelvis_mass_height, 0.16f, t.pelvis_density);
    setSphereMass(&bodies[B_ROLLER], kRollerRadius, t.roller_density);  // m1 = d1 (m1 arg 0)
    setSphereMass(&bodies[B_STAND], 0.3f, 1000.0f);
    roller_geom_radius = kRollerRadius;
    stand(0, 0, 0, 0);
    alljoints.clear();
    Joint* arr = joints; memset(joints, 0, sizeof(joints));
    auto mkfixed = [&](int idx, Body* b1, Body* b2, R ls, R ld, R as, R ad) -> Joint* {
      Joint* j = &arr[idx]; j->type = J_FIXEDEF; attach(j, b1, b2); qMul1(j->qrel, b1->q, b2->q);
      R q[4]; q[0] = b2->pos[0] - b1->pos[0]; q[1] = b2->pos[1] - b1->pos[1]; q[2] = b2->pos[2] - b1->pos[2]; q[3] = 0;
      mul1_331(j->a1, b1->Rm, q);
      q[0] = b2->pos[0] - b2->pos[0]; q[1] = b2->pos[1] - b2->pos[1]; q[2] = b2->pos[2] - b2->pos[2]; q[3] = 0;
      mul1_331(j->a2, b2->Rm, q); j->a1[3] = 0; j->a2[3] = 0;
      j->ls = ls; j->ld = ld; j->as = as; j->ad = ad; alljoints.push_back(j); return j; };
    neck = mkfixed(0, &bodies[B_HEAD], &bodies[B_TORSO], 1000, 1, 20.0f, 0.3f);
    neck->a1[1] += 0.2f; neck->a2[1] += 0.2f;
    pelvisJ = mkfixed(1, &bodies[B_PELVIS], &bodies[B_TORSO], 0, 0, 0, 0);
    pelvisJ->a1[1] -= kPelvisJointDrop; pelvisJ->a2[1] -= kPelvisJointDrop; pelvisJ->a2[2] += kPelvisJointForward;
    // UpdateJoints() -> ApplyRestPose for the pelvis (not frozen)
    qFromAxisAngle(pelvisJ->qrel, 1, 0.0f, 0.0f, -0.4f);
    pelvisJ->ls = t.pelvis_linear_stiffness; pelvisJ->ld = t.pelvis_linear_damping;
    pelvisJ->as = t.pelvis_angular_stiffness; pelvisJ->ad = t.pelvis_angular_damping;
    rollerJ = mkfixed(2, &bodies[B_TORSO], &bodies[B_ROLLER], t.roller_joint_stiffness, t.roller_joint_damping, 0, 0);
    base_roller_anchor = rollerJ->a1[1];
    standJ = mkfixed(3, &bodies[B_TORSO], &bodies[B_STAND], 100, 1, 200, 10);
    motorRoller = &arr[4]; motorRoller->type = J_AMOTOR; attach(motorRoller, &bodies[B_ROLLER], nullptr);
    motorBrakes = &arr[5]; motorBrakes->type = J_AMOTOR; attach(motorBrakes, &bodies[B_TORSO], &bodies[B_ROLLER]);
    for (Joint* m : {motorRoller, motorBrakes}) { m->num = 3;
      for (int i = 0; i < 3; i++) { m->ncfm[i] = kGlobalCFM; m->fmax[i] = 0; m->vel[i] = 0; } }
    for (int i = 0; i < 3; i++) { motorRoller->rel[i] = 0; motorBrakes->rel[i] = 1;
      memset(motorRoller->axis[i], 0, 16); memset(motorBrakes->axis[i], 0, 16);
      motorRoller->axis[i][i] = 1; motorBrakes->axis[i][i] = 1; }
    motorRoller->fmax[0] = motorRoller->fmax[1] = motorRoller->fmax[2] = 3.0f; motorRoller->vel[2] = 1.0f;
    motorBrakes->fmax[0] = motorBrakes->fmax[1] = motorBrakes->fmax[2] = 10.0f;
    alljoints.push_back(motorRoller); alljoints.push_back(motorBrakes);
  }
  void stand(R x, R y, R z, R angle) {
    y -= 0.7f; R iq[4]; qFromAxisAngle(iq, 0, 1, 0, angle * (3.14159265358979323846f / 180.0f));
    R hs[5] = {kStandHead, kStandTorso, kStandPelvis, kStandRoller, kStandTorso};
    for (int i = 0; i < 5; i++) { Body* b = &bodies[i];
      b->pos[0] = x; b->pos[1] = y + hs[i]; b->pos[2] = z; memset(b->lvel, 0, 16); memset(b->avel, 0, 16);
      setQuat(b, iq); memset(b->facc, 0, 16); memset(b->tacc, 0, 16); }
  }
  void reset(R x, R y, R z, R angle) {
    *this = Sim(); build(); stand(x, y, z, angle);
  }

  // ---- terrain collision (analytic boxes; sphere-vs-box is ODE's exact algorithm) ----
  int sphereBox(const R* sp, R radius, const Box& bx, Contact* out) {
    R pos[3], l[3], t[3], p[3];
    for (int i = 0; i < 3; i++) { pos[i] = (bx.mn[i] + bx.mx[i]) * 0.5f; l[i] = (bx.mx[i] - bx.mn[i]) * 0.5f; p[i] = sp[i] - pos[i]; t[i] = p[i]; }
    bool onborder = false;
    for (int i = 0; i < 3; i++) { if (t[i] < -l[i]) { t[i] = -l[i]; onborder = true; } if (t[i] > l[i]) { t[i] = l[i]; onborder = true; } }
    if (!onborder) {
      R md = l[0] - fabsf(t[0]); int mini = 0;
      for (int i = 1; i < 3; i++) { R fd = l[i] - fabsf(t[i]); if (fd < md) { md = fd; mini = i; } }
      out->pos[0] = sp[0]; out->pos[1] = sp[1]; out->pos[2] = sp[2];
      out->normal[0] = out->normal[1] = out->normal[2] = 0; out->normal[mini] = (t[mini] > 0) ? 1.0f : -1.0f;
      out->depth = md + radius; return 1;
    }
    R r[3]; for (int i = 0; i < 3; i++) r[i] = p[i] - t[i];
    R depth = radius - sqrtf(dot3(r, r)); if (depth < 0) return 0;
    for (int i = 0; i < 3; i++) { out->pos[i] = t[i] + pos[i]; out->normal[i] = r[i]; }
    normalize3(out->normal); out->depth = depth; return 1;
  }
  // box (pelvis) vs the top face of a map box: ODE dCollideBoxPlane contact selection, plane = top face.
  int boxTop(const Body* pb, const Box& bx, Contact* out) {
    R top = bx.mx[1]; const R* Rm = pb->Rm; R n[3] = {0, 1, 0};
    if (pb->pos[0] < bx.mn[0] || pb->pos[0] > bx.mx[0] || pb->pos[2] < bx.mn[2] || pb->pos[2] > bx.mx[2]) return 0;
    R side[3] = {kPelvisSize[0], kPelvisSize[1], kPelvisSize[2]};
    R Q1 = n[0] * Rm[0] + n[1] * Rm[4] + n[2] * Rm[8], Q2 = n[0] * Rm[1] + n[1] * Rm[5] + n[2] * Rm[9], Q3 = n[0] * Rm[2] + n[1] * Rm[6] + n[2] * Rm[10];
    R A[3] = {side[0] * Q1, side[1] * Q2, side[2] * Q3}; R B[3] = {fabsf(A[0]), fabsf(A[1]), fabsf(A[2])};
    R depth = top + 0.5f * (B[0] + B[1] + B[2]) - dot3(n, pb->pos); if (depth < 0) return 0;
    R p[3] = {pb->pos[0], pb->pos[1], pb->pos[2]};
    for (int i = 0; i < 3; i++) { R s = 0.5f * side[i];
      if (A[i] > 0) { p[0] -= s * Rm[0 + i]; p[1] -= s * Rm[4 + i]; p[2] -= s * Rm[8 + i]; }
      else { p[0] += s * Rm[0 + i]; p[1] += s * Rm[4 + i]; p[2] += s * Rm[8 + i]; } }
    int ret = 0; memcpy(out[0].pos, p, 12); out[0].depth = depth; ret = 1;
    int s1, s2;  // sides in increasing |A| order per ODE's branch structure
    if (B[0] < B[1]) { if (B[2] < B[0]) { s1 = 2; s2 = (B[0] < B[1]) ? 0 : 1; } else { s1 = 0; s2 = (B[1] < B[2]) ? 1 : 2; } }
    else { if (B[2] < B[1]) { s1 = 2; s2 = (B[0] < B[1]) ? 0 : 1; } else { s1 = 1; s2 = (B[0] < B[2]) ? 0 : 2; } }
    int sides[2] = {s1, s2};
    for (int k = 0; k < 2; k++) { int sd = sides[k]; depth -= B[sd]; if (depth < 0) break;
      R sg = (A[sd] > 0) ? 1.0f : -1.0f;
      out[ret].pos[0] = p[0] + sg * side[sd] * Rm[0 + sd]; out[ret].pos[1] = p[1] + sg * side[sd] * Rm[4 + sd]; out[ret].pos[2] = p[2] + sg * side[sd] * Rm[8 + sd];
      out[ret].depth = depth; ret++; }
    for (int i = 0; i < ret; i++) { out[i].normal[0] = n[0]; out[i].normal[1] = n[1]; out[i].normal[2] = n[2]; }
    return ret;
  }
  static void erpcfm(R s, R d, R* erp, R* cfm) { calcERPCFM(s, d, kStep, erp, cfm); }

  void collideBody(int bi) {
    Body* b = &bodies[bi]; if (steps_age * kStepMs < 51) return;  // 'we_are_younger_than 51'
    Contact cs[24]; int terrain_contacts = 0;
    for (const Box& bx : map) {
      int n = 0;
      if (bi == B_PELVIS) n = boxTop(b, bx, cs);
      else { R rad = (bi == B_HEAD) ? kHeadRadius : (bi == B_TORSO ? torso_radius : roller_geom_radius);
        n = sphereBox(b->pos, rad, bx, cs); }
      if (n <= 0) continue;
      terrain_contacts += n;
      // default material combine (friction/bounce/stiffness/damping of two default contexts)
      R friction = 1.2f * sqrtf(1.0f * 1.0f), bounce = sqrtf(0.0f * 0.0f);
      R stiffness = 8000 * sqrtf(1.0f * 1.0f), damping = 80 * 1.0f + 1.0f;
      R erp = (kStep * stiffness) / ((kStep * stiffness) + damping), cfm = 1.0f / ((kStep * stiffness) + damping);
      for (int i = 0; i < n; i++) { cs[i].mu = friction; cs[i].bounce = bounce; cs[i].bounce_vel = 0.1f; cs[i].soft_cfm = cfm; cs[i].soft_erp = erp; }
      // SpazNode::CollideCallback
      if (bi != B_ROLLER) { R e, c; erpcfm(t.core_contact_stiffness, t.core_contact_damping, &e, &c);
        for (int i = 0; i < n; i++) { cs[i].soft_erp = e; cs[i].soft_cfm = c; } }
      if (bi == B_ROLLER) {
        bool down_ball = (knockout != 0);
        for (int i = 0; i < n; i++) {
          R dotv = fabsf(cs[i].normal[0] * 0 + cs[i].normal[1] * 1 + cs[i].normal[2] * 0);
          if (dotv > 1) dotv = 1; else if (dotv < 0) dotv = 0;
          if (dotv < 0.6f) {
            if (!down_ball) { R kick = t.roller_wall_kick_force;
              b->facc[0] += cs[i].normal[0] * kick; b->facc[1] += cs[i].normal[1] * kick; b->facc[2] += cs[i].normal[2] * kick; }
            R e, c; erpcfm(t.roller_side_contact_stiffness, t.roller_side_contact_damping, &e, &c);
            cs[i].soft_erp = e; cs[i].soft_cfm = c; cs[i].mu = 0.0f;
          } else {
            R st = down_ball ? t.down_ball_contact_stiffness : t.roller_floor_contact_stiffness;
            R dp = down_ball ? t.down_ball_contact_damping : t.roller_floor_contact_damping;
            R e, c; erpcfm(st, dp, &e, &c); cs[i].soft_erp = e; cs[i].soft_cfm = c; cs[i].mu *= 1.0f;
          }
        }
        if (wake_slide > 0) for (int i = 0; i < n; i++) cs[i].mu = 0.0f;
      } else for (int i = 0; i < n; i++) cs[i].mu *= t.body_friction_scale;
      for (int i = 0; i < n; i++) {
        contacts.emplace_back(); contact_body.push_back(bi); Joint& j = contacts.back(); memset(&j, 0, sizeof(j)); j.type = J_CONTACT; j.c = cs[i];
      }
    }
    (void)terrain_contacts;
  }

  // one SpazNode::Step()
  void spazStep() {
    Body *head = &bodies[B_HEAD], *torso = &bodies[B_TORSO], *pelvis = &bodies[B_PELVIS], *roller = &bodies[B_ROLLER], *stand_b = &bodies[B_STAND];
    (void)pelvis;
    R scenetime_ms = (R)(stepnum * kStepMs);
    // smoothed stick values
    { R smoothing;
      if ((ud == -127 || ud == 0 || ud == 127) && (lr == -127 || lr == 0 || lr == 127)) smoothing = 0.5f; else smoothing = 0.0f;
      ud_smooth = smoothing * ud_smooth + (1.0f - smoothing) * ((R)ud / 127.0f);
      lr_smooth = smoothing * lr_smooth + (1.0f - smoothing) * ((R)lr / 127.0f); }
    { R prev_ud = ud_norm, prev_lr = lr_norm;
      R this_ud = ((R)ud / 127.0f), this_lr = ((R)lr / 127.0f);
      { R len2 = this_lr * this_lr + this_ud * this_ud; if (len2 > 1.0f) { R len = sqrtf(len2); R mult = 1.0f / len; this_lr *= mult; this_ud *= mult; } }
      running = ((run > 0.0f) && (abs((int)lr) > 0 || abs((int)ud) > 0));
      if (running) { R run_target = sqrtf(run); R mag = (lr_smooth * lr_smooth + ud_smooth * ud_smooth);
        if (mag < 0.3f) run_target *= (mag / 0.3f);
        R smoothing = run_target > run_gas ? 0.95f : 0.5f; run_gas = smoothing * run_gas + (1.0f - smoothing) * run_target;
      } else run_gas = std::max(0.0f, run_gas - 0.02f);
      if (!footing) run_gas = std::max(0.0f, run_gas - 0.05f);
      if (run > 0.05f) {
        { R cur_dir[3] = {ud_norm, lr_norm, 0}, new_dir[3] = {this_ud, this_lr, 0}; R d = dot3(new_dir, cur_dir);
          if (d < 0.0f) { this_ud -= run_gas * (ud_norm * d); this_lr -= run_gas * (lr_norm * d);
            if (this_ud == 0.0f) this_ud = -0.001f; if (this_lr == 0.0f) this_lr = -0.001f; } }
        R orig_len, target_len; R ud_nn = this_ud, lr_nn = this_lr;
        { orig_len = sqrtf(ud_nn * ud_nn + lr_nn * lr_nn); target_len = run_gas * 1.0f + (1.0f - run_gas) * orig_len;
          R mult = orig_len == 0.0f ? 1.0f : target_len / orig_len; ud_nn *= mult; lr_nn *= mult; }
        R v[3] = {torso->lvel[0], torso->lvel[1], torso->lvel[2]}; R speed = vlen(v);
        R smoothing = 0.975f * (0.9f + 0.1f * run_gas); if (speed < 2.0f) smoothing *= (speed / 2.0f);
        ud_norm = smoothing * ud_norm + (1.0f - smoothing) * ud_nn; lr_norm = smoothing * lr_norm + (1.0f - smoothing) * lr_nn;
        R new_len = sqrtf(ud_norm * ud_norm + lr_norm * lr_norm); R mult = new_len == 0.0f ? 1.0f : target_len / new_len;
        ud_norm *= mult; lr_norm *= mult;
      } else { ud_norm = this_ud; lr_norm = this_lr; }
      R sd = 0.93f; ud_diff_smooth = sd * ud_diff_smooth + (1.0f - sd) * (ud_norm - prev_ud); lr_diff_smooth = sd * lr_diff_smooth + (1.0f - sd) * (lr_norm - prev_lr);
      R sr = 0.983f; ud_diff_smoother = sr * ud_diff_smoother + (1.0f - sr) * (ud_norm - prev_ud); lr_diff_smoother = sr * lr_diff_smoother + (1.0f - sr) * (lr_norm - prev_lr);
    }
    { R avel = torso->avel[1]; R s = 0.7f; a_vel_y_smoothed = s * a_vel_y_smoothed + (1.0f - s) * avel;
      s = 0.92f; a_vel_y_smoothed_more = s * a_vel_y_smoothed_more + (1.0f - s) * avel; }
    if (footing) fly_time = 0; else fly_time++;
    // velocity clamps (head, torso)
    for (Body* b : {head, torso}) {
      R ms = b->avel[0] * b->avel[0] + b->avel[1] * b->avel[1] + b->avel[2] * b->avel[2];
      if (ms > 400.0f) { R sc = 400.0f / ms; b->avel[0] = b->avel[0] * sc; b->avel[1] = b->avel[1] * sc; b->avel[2] = b->avel[2] * sc; }
      ms = b->lvel[0] * b->lvel[0] + b->lvel[1] * b->lvel[1] + b->lvel[2] * b->lvel[2];
      if (ms > 300.0f) { R sc = 300.0f / ms; b->lvel[0] = b->lvel[0] * sc; b->lvel[1] = b->lvel[1] * sc; b->lvel[2] = b->lvel[2] * sc; }
    }
    R breath = 0.0f;
    if (!lr && !ud) breath = sinf(scenetime_ms * 0.005f);
    // ---- SpazPose::Step: neck strength/target and roll_amt ----
    if (knockout) { neck->ls = t.neck_ko_linear_stiffness; neck->ld = t.neck_ko_linear_damping; neck->as = t.neck_ko_angular_stiffness; neck->ad = t.neck_ko_angular_damping; }
    else { neck->ls = t.neck_linear_stiffness; neck->ld = t.neck_linear_damping; neck->as = t.neck_angular_stiffness; neck->ad = t.neck_angular_damping; }
    if (knockout || balance == 0) {
      if (!footing && balance == 0) { roll_amt -= 0.2f; if (roll_amt < (-2.0f * 3.141592f)) roll_amt += 2.0f * 3.141592f; }
    } else {
      const R* ba = roller->avel; R aVelMag = sqrtf(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]);
      R rollScale = 1.0f; rollScale = run_gas * 0.8f + (1.0f - run_gas) * rollScale;
      roll_amt -= rollScale * 0.021f * std::max(aVelMag - 0.1f, 0.0f);
      if (roll_amt < (-2.0f * 3.141592f)) roll_amt += 2.0f * 3.141592f;
    }
    { int64_t gti = stepnum;
      if ((ud || lr) || knockout) {
        qFromAxisAngle(neck->qrel, 1, 0, 0, 0.0f); neck->qrel[0] = 1; neck->qrel[1] = neck->qrel[2] = neck->qrel[3] = 0;  // identity
        qFromAxisAngle(neck->qrel, 0, 1, 0, std::max(-1.0f, std::min(1.0f, a_vel_y_smoothed_more * -0.14f)));
      } else if (gti % 30 == 0 && kPrecalc1[(gti + 0 * 3 + 143) % 128] > 0.9f) {
        qFromAxisAngle(neck->qrel, kPrecalc1[(0 + gti) % (128 - 3)] * 0.05f, kPrecalc2[(42 * gti) % 128], kPrecalc3[(3 * gti) % (128 - 1)] * 0.05f,
                       1.5f * (kPrecalc2[(gti) % 128] - 0.5f));
      } }
    // dizzy -> knockout (behavior_version >= 2 in normal play)
    if (behavior_version > 0) {
      if (fabsf(a_vel_y_smoothed_more) > 10.0f) { dizzy += 1; if (dizzy > 120) { dizzy = 0; knockout = 40; } }
      else dizzy = (uint8_t)std::max(0, (int)dizzy - 2);
    }
    if (knockout > 0) balance = 0;
    else { if (footing) { if (balance < 100) balance += 20; else if (balance < 235) balance += 20; else if (balance < 255) balance++; }
      else { if (balance > 100) balance -= 20; else if (balance > 10) balance -= 5; else if (balance > 0) balance--; } }
    if (knockout > 0 && (stepnum % (footing ? 5 : 10) == 0)) { knockout--; if (knockout == 0) wake_slide = (uint8_t)t.wake_slide_steps; }
    else if (wake_slide > 0) wake_slide--;
    // ---- stand body + stand joint ----
    { Body* b = stand_b; const R* p_torso2 = torso->pos; const R* p_bot = roller->pos; const R* lv = torso->lvel;
      memcpy(b->lvel, lv, 12); b->avel[0] = b->avel[1] = b->avel[2] = 0;
      R tilt_lr, tilt_ud; b->pos[0] = p_torso2[0]; b->pos[1] = p_bot[1] + 0.2f; b->pos[2] = p_torso2[2];
      R rotate_tilt = t.lean_amount;
      { const R* bv = roller->lvel; R vm = std::max(7.0f, vlen(bv)); R as = 0.7f;
        for (int i = 0; i < 3; i++) { R av = bv[i]; accel[i] = as * accel[i] + (1.0f - as) * (av - prev_vel[i]); prev_vel[i] = av; }
        tilt_lr = (0.2f + 0.8f * run_gas) * std::min(0.9f, std::max(-0.9f, vm * accel[0] * 0.3f));
        tilt_ud = (0.2f + 0.8f * run_gas) * std::min(0.9f, std::max(-0.9f, vm * accel[2] * -0.3f));
        R fast = std::min(1.0f, speed_smoothed / 5.0f);
        tilt_lr += (1.0f - fast) * (lr_diff_smooth * 10.0f); tilt_ud += (1.0f - fast) * (ud_diff_smooth * 10.0f);
        tilt_lr += fast * (lr_diff_smoother * 30.0f); tilt_ud += fast * (ud_diff_smoother * 30.0f);
        rotate_tilt *= 1.2f; }
      R spin = fabsf(torso->avel[1]); if (spin > 10.0f) rotate_tilt = 0.0f;
      auto rotFrom2Axes = [&](R* r, R xf, R yf, R zf, R xu, R yu, R zu) {
        R fwd[3] = {xf, yf, zf}, up[3] = {xu, yu, zu}; R ul = vlen(up); if (ul > 0) { R inv = 1.0f / ul; up[0] *= inv; up[1] *= inv; up[2] *= inv; }
        R side[3]; cross(side, fwd, up); R sl = vlen(side); if (sl > 0) { R inv = 1.0f / sl; side[0] *= inv; side[1] *= inv; side[2] *= inv; }
        R f2[3]; cross(f2, up, side);
        r[0] = f2[0]; r[4] = f2[1]; r[8] = f2[2]; r[1] = up[0]; r[5] = up[1]; r[9] = up[2]; r[2] = side[0]; r[6] = side[1]; r[10] = side[2]; };
      if (lr || ud) { R r[12] = {0};
        rotFrom2Axes(r, -ud_norm, 0, -lr_norm, rotate_tilt * tilt_lr, 1, -rotate_tilt * tilt_ud); setRot(b, r);
      } else { R r[12] = {0}; R pf[3]; { R prel[3] = {1, 0, 0}, pp[3]; mul0_331(pp, torso->Rm, prel); pf[0] = pp[0] + torso->pos[0]; pf[1] = pp[1] + torso->pos[1]; pf[2] = pp[2] + torso->pos[2]; }
        R ox = pf[0] - p_torso2[0], oz = pf[2] - p_torso2[2];
        if (fabsf(ox - last_orient_x) > 0.05f || fabsf(oz - last_orient_z) > 0.05f) { last_orient_x = ox; last_orient_z = oz; }
        rotFrom2Axes(r, last_orient_x, 0, last_orient_z, rotate_tilt * tilt_lr, 1, -rotate_tilt * tilt_ud); setRot(b, r); }
      { R angle = sinf(roll_amt - 3.141592f) * (run_gas * 0.09f + (1.0f - run_gas) * idle_sway); qFromAxisAngle(standJ->qrel, 0, 1, 1, angle); }
      { R bal = (R)balance / 255.0f; bal = 1.0f - ((1.0f - bal) * (1.0f - bal) * (1.0f - bal) * (1.0f - bal)); R mult = bal; mult *= t.stand_balance_scale;
        standJ->ls = 0.0f; standJ->ld = 0.0f; standJ->as = t.stand_angular_stiffness * mult; standJ->ad = t.stand_angular_damping * mult;
        R f[3] = {ud_norm, 0, lr_norm}; R m = vlen(f); R blend_max = 1.0f;
        if (m < blend_max) { standJ->ad *= 0.3f + 0.7f * (m / blend_max); standJ->as *= 0.6f + 0.4f * (m / blend_max); } }
    }
    // ---- roller size / suspension spring ----
    { if (knockout) ball_size = std::min(t.down_ball_size, ball_size + 0.05f); else ball_size = std::min(1.0f, ball_size + 0.05f);
      R sz = 0.1f + 0.9f * ball_size; roller_geom_radius = 0.3f * sz; setSphereMass(roller, 0.3f, t.roller_density); }
    { R offs = (1.0f - ball_size) * 0.3f;
      R r_down = 0.3f * (0.1f + 0.9f * t.down_ball_size); R offs_down = t.down_ball_bottom_lift - (0.3f - r_down);
      R tt = std::min(1.0f, std::max(0.0f, (1.0f - ball_size) / (1.0f - t.down_ball_size))); offs = tt * offs_down;
      R ls_scale = 1.0f, ld_scale = 1.0f;
      if (jump > 0 && !knockout) { offs -= t.jump_push_distance; ls_scale = t.jump_roller_stiffness_scale; ld_scale = t.jump_roller_damping_scale; }
      rollerJ->ls = t.roller_joint_stiffness * ls_scale; rollerJ->ld = t.roller_joint_damping * ld_scale;
      offs -= breath * 0.02f; rollerJ->a1[1] = base_roller_anchor + offs; }
    // ---- roller motor ----
    { R mult = std::min(1.0f, (R)balance / 100.0f);
      R v[3] = {roller->lvel[0], roller->lvel[1], roller->lvel[2]}; R speed_scale = 1.0f, walk_scale = 1.0f;
      if (footing) { if (v[1] < 0.0f) { v[1] *= 2.0f; walk_scale = 1.0f - v[1] * t.downhill_boost; }
        else { speed_scale = std::max(0.0f, 1.0f - v[1] * t.uphill_slow); walk_scale = std::max(0.0f, 1.0f - v[1] * t.uphill_slow); v[1] = 0.0f; } }
      R speed = vlen(v) * speed_scale; R ss = (speed > speed_smoothed) ? t.speed_smoothing_up : t.speed_smoothing_down;
      speed_smoothed = ss * speed_smoothed + (1.0f - ss) * speed;
      R gear_high = std::min(1.0f, speed_smoothed / t.gear_up_speed), gear_low = 1.0f - gear_high;
      R max_force = gear_low * t.roller_motor_max_force_low_gear + gear_high * t.roller_motor_max_force_high_gear;
      R max_vel = walk_scale * t.walk_speed + gear_high * run_gas * t.run_speed_add;
      motorRoller->fmax[0] = max_force * mult; motorRoller->fmax[1] = t.roller_motor_max_force_vertical * mult; motorRoller->fmax[2] = max_force * mult;
      motorRoller->vel[0] = -max_vel * ud_norm; motorRoller->vel[1] = 0.0f; motorRoller->vel[2] = -max_vel * lr_norm; }
    { bool down_brakes = t.down_ball_brakes && knockout > 0;
      R bf = 0.0f;
      if (footing || down_brakes) { R amt;
        if (down_brakes) amt = 1.0f; else { R f[3] = {lr_norm, 0, ud_norm}; amt = std::min(1.0f, vlen(f) * 5.0f); amt = 1.0f - (amt * amt * amt); amt *= (1.0f - run_gas); amt *= t.brake_idle_scale; }
        bf = t.brake_max_force * amt; }
      for (int i = 0; i < 3; i++) { motorBrakes->fmax[i] = bf; motorBrakes->vel[i] = 0.0f; } }
    if (jump > 0) jump -= 1;
  }

  // Scene::Step: node Step -> collisions (footing) -> world step
  void step(R lrv, R udv, R runv, bool jump_in) {
    // attribute setters (SetMoveLeftRight etc.)
    lr = (int8_t)std::max(-127, std::min(127, (int)(127.0f * lrv)));
    ud = (int8_t)std::max(-127, std::min(127, (int)(127.0f * udv)));
    run = runv;
    if (jump_in != jump_pressed) { jump_pressed = jump_in; if (jump_pressed && !knockout) jump = (uint8_t)t.jump_steps; }
    spazStep();
    // collision pass: terrain geoms in hash-space order (stand, roller, pelvis, torso, head)
    contacts.clear(); contact_body.clear();
    collideBody(B_ROLLER); size_t roller_n = contacts.size();
    collideBody(B_PELVIS); collideBody(B_TORSO); collideBody(B_HEAD);
    size_t total_n = contacts.size();
    // footing = number of (our part, terrain) contact pairs: roller part + spaz part
    footing = (roller_n > 0 ? 1 : 0) + (total_n > roller_n ? 1 : 0);
    { size_t k = 0; for (size_t i = 0; i < contacts.size(); i++) attach(&contacts[i], &bodies[contact_body[i]], nullptr); (void)k; }
    Body* list[5] = {&bodies[B_STAND], &bodies[B_ROLLER], &bodies[B_PELVIS], &bodies[B_TORSO], &bodies[B_HEAD]};  // newest first
    std::vector<Joint*> all = alljoints; for (Joint& c : contacts) all.push_back(&c);
    processIslands(world, list, 5, all);
    for (size_t i = contacts.size(); i-- > 0;) detachContact(&contacts[i]);
    steps_age++; stepnum++;
  }
  std::vector<int> contact_body;
};

// ------------------------------------------------------------- C API ----
// (see spazsim_api.cpp section below)

extern "C" {
Sim* ss_create() { Sim* s = new Sim(); s->build(); return s; }
void ss_destroy(Sim* s) { delete s; }
void ss_set_map(Sim* s, const float* boxes, int n) {  // n boxes of (minx,miny,minz,maxx,maxy,maxz)
  s->map.clear(); for (int i = 0; i < n; i++) { Box b; for (int k = 0; k < 3; k++) { b.mn[k] = boxes[i*6+k]; b.mx[k] = boxes[i*6+3+k]; } s->map.push_back(b); }
}
void ss_reset(Sim* s, float x, float y, float z, float angle) {
  std::vector<Box> m = s->map; s->reset(x, y, z, angle); s->map = m;
}
void ss_step(Sim* s, float lr, float ud, float run, int jump) { s->step(lr, ud, run, jump != 0); }
// state: 0..2 torso pos, 3..5 torso vel, 6..8 torso avel, 9..11 torso up vector, 12..14 roller pos,
// 15..17 roller vel, 18..20 head pos, 21 balance/255, 22 footing, 23 knockout, 24 run_gas, 25 ball_size
void ss_state(const Sim* s, float* o) {
  const Body* t = &s->bodies[B_TORSO]; const Body* r = &s->bodies[B_ROLLER]; const Body* h = &s->bodies[B_HEAD];
  for (int i = 0; i < 3; i++) { o[i] = t->pos[i]; o[3+i] = t->lvel[i]; o[6+i] = t->avel[i]; o[9+i] = t->Rm[4*i+1];
    o[12+i] = r->pos[i]; o[15+i] = r->lvel[i]; o[18+i] = h->pos[i]; }
  o[21] = s->balance / 255.0f; o[22] = (float)s->footing; o[23] = (float)s->knockout; o[24] = s->run_gas; o[25] = s->ball_size;
}
}
