const L0 = 1;
const L1 = 1;

function calculatePs(theta, s, alpha) {
    const term1 = [
        L0 * Math.cos(theta),
        L0 * Math.sin(theta),
        0
    ];

    const term2 = [
        s * Math.sin(alpha) * Math.sin(theta),
        -s * Math.sin(alpha) * Math.cos(theta),
        s * Math.cos(alpha)
    ];

    return [
        term1[0] + term2[0],
        term1[1] + term2[1],
        term1[2] + term2[2]
    ];
}

function createTraces(theta, alpha) {
    const origin = [0, 0, 0];
    const armEnd = calculatePs(theta, 0, alpha);
    const pendulumEnd = calculatePs(theta, L1, alpha);

    return [
        // axes
        {
            type: "scatter3d",
            mode: "lines",
            x: [0, 1], y: [0, 0], z: [0, 0],
            line: { color: "white", width: 2 },
            showlegend: false
        },
        {
            type: "scatter3d",
            mode: "lines",
            x: [0, 0], y: [0, 1], z: [0, 0],
            line: { color: "white", width: 2 },
            showlegend: false
        },
        {
            type: "scatter3d",
            mode: "lines",
            x: [0, 0], y: [0, 0], z: [0, 1],
            line: { color: "white", width: 2 },
            showlegend: false
        },

        // arm
        {
            type: "scatter3d",
            mode: "lines",
            x: [origin[0], armEnd[0]],
            y: [origin[1], armEnd[1]],
            z: [origin[2], armEnd[2]],
            line: { color: "lightblue", width: 4 },
            showlegend: false
        },

        // pendulum
        {
            type: "scatter3d",
            mode: "lines",
            x: [armEnd[0], pendulumEnd[0]],
            y: [armEnd[1], pendulumEnd[1]],
            z: [armEnd[2], pendulumEnd[2]],
            line: { color: "lightblue", width: 4 },
            showlegend: false
        },

        // joint
        {
            type: "scatter3d",
            mode: "markers",
            x: [armEnd[0]],
            y: [armEnd[1]],
            z: [armEnd[2]],
            marker: { color: "white", size: 4 },
            showlegend: false
        }
    ];
}


function createAnimation(thetaFrames, alphaFrames) {
    const frames = [];

    for (let i = 0; i < thetaFrames.length; i++) {
        frames.push({
            name: `frame-${i}`,
            data: createTraces(thetaFrames[i], alphaFrames[i])
        });
    }

    return frames;
}
