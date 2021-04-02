OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
