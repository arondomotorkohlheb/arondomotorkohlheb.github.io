// qube_dynamics.js
// Standalone Qube dynamics module

export class Qube {
    /**
     * x = [theta_dot, alpha_dot, theta, alpha]
     * dt = time step
     * K = hard-coded control matrix
     */
    constructor(parameters, dt=0.02) {
        this.params = Object.assign({
            L0: 1,
            L1: 1,
            gravity: 9.81,
            zeta_theta: 0.1,
            zeta_alpha: 0.1,
            v2tau: 1.0,
            k0: 0.01
            r: 0.085 / 0.129,
            k0: 0.276139274,
            zeta_theta: 1.96602143,
            zeta_alpha: 0.301178062,
            v2tau: 440.092256,
            gravity: 9.81 / (2 * 0.129),
            L0: 0.085,
            L1: 0.129,
        }, parameters);

        this.dt = dt;

        // state: [theta_dot, alpha_dot, theta, alpha]
        this.x = [0, 0, 0, 0];
        this.xref = [0, 0, 0, 0];

        // hard-coded LQR gain matrix
        this.K = [[-0.09177338, 0.13273703, -0.31622777, 1.57357494]];
    }

    // Compute control input u
    controlInput() {
        const dx = this.xref.map((v,i)=>v - this.x[i]);
        let u = 0;
        for(let i=0;i<4;i++) u += this.K[0][i]*dx[i];
        return u;
    }

    // Compute armEnd and pendulumEnd positions
    computePositions() {
        const theta = this.x[2];
        const alpha = this.x[3];
        const armEnd = {
            x: this.params.L0*Math.cos(theta),
            y: this.params.L0*Math.sin(theta),
            z: 0
        };
        const dir = {
            x: Math.sin(alpha)*Math.sin(theta),
            y: -Math.sin(alpha)*Math.cos(theta),
            z: Math.cos(alpha)
        };
        const pendulumEnd = {
            x: armEnd.x + this.params.L1*dir.x,
            y: armEnd.y + this.params.L1*dir.y,
            z: armEnd.z + this.params.L1*dir.z
        };
        return {armEnd, pendulumEnd};
    }

    // Compute accelerations
    computeAcceleration(u) {
        const [theta_dot, alpha_dot, theta, alpha] = this.x;

        const C0 = [-Math.sin(theta), Math.cos(theta), 0];
        const C1 = [Math.sin(alpha)*Math.cos(theta), Math.sin(alpha)*Math.sin(theta), 0];
        const C2 = [Math.cos(alpha)*Math.sin(theta), -Math.cos(alpha)*Math.cos(theta), -Math.sin(alpha)];

        const r = this.params.L0 / this.params.L1;
        const M11 = r*r*(C0[0]**2+C0[1]**2+C0[2]**2) + r*(C0[0]*C1[0]+C0[1]*C1[1]+C0[2]*C1[2]) + (1/3)*(C1[0]**2+C1[1]**2+C1[2]**2);
        const M12 = 0.5*r*(C0[0]*C2[0]+C0[1]*C2[1]+C0[2]*C2[2]) + (1/3)*(C1[0]*C2[0]+C1[1]*C2[1]+C1[2]*C2[2]);
        const M22 = (1/3)*(C2[0]**2+C2[1]**2+C2[2]**2);

        const det = M22*(M11+this.params.k0) - M12*M12;
        const Mbar_inv = [
            [ M22/det, -M12/det],
            [-M12/det, (M11+this.params.k0)/det ]
        ];

        const tau = [this.params.v2tau*u - this.params.zeta_theta*theta_dot, -this.params.zeta_alpha*alpha_dot];
        const gravity = [0, this.params.gravity*Math.sin(alpha)];

        // Malpha term approximation (simplified)
        const Malpha = [0,0];

        const acc0 = Mbar_inv[0][0]*(tau[0]+gravity[0]+Malpha[0]) + Mbar_inv[0][1]*(tau[1]+gravity[1]+Malpha[1]);
        const acc1 = Mbar_inv[1][0]*(tau[0]+gravity[0]+Malpha[0]) + Mbar_inv[1][1]*(tau[1]+gravity[1]+Malpha[1]);

        return [acc0, acc1];
    }

    // Integrate the state by one timestep
    step() {
        const u = this.controlInput();
        const [acc_theta, acc_alpha] = this.computeAcceleration(u);

        this.x[0] += acc_theta * this.dt;
        this.x[1] += acc_alpha * this.dt;
        this.x[2] += this.x[0] * this.dt;
        this.x[3] += this.x[1] * this.dt;

        return this.x;
    }

    // Get current theta and alpha angles
    getAngles() {
        return {theta: this.x[2], alpha: this.x[3]};
    }
}