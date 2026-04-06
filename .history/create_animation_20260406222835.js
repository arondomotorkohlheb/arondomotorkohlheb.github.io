// REMAP FUNCTION
// input: THREE.Vector3(vx,vy,vz) in your original convention
// output: remapped vector in Three.js default axes
function remap(v){
    // Example: newX = oldY, newY = oldZ, newZ = oldX
    return new THREE.Vector3(v.y, v.z, v.x);
}

// USAGE in compute():
function compute(theta, alpha){
    const armEndOrig = new THREE.Vector3(
        L0*Math.cos(theta),
        L0*Math.sin(theta),
        0
    );

    const dirOrig = new THREE.Vector3(
        Math.sin(alpha)*Math.sin(theta),
        -Math.sin(alpha)*Math.cos(theta),
        Math.cos(alpha)
    );

    const pendulumEndOrig = armEndOrig.clone().addScaledVector(dirOrig, L1);

    // remap all
    const armEnd = remap(armEndOrig);
    const pendulumEnd = remap(pendulumEndOrig);

    return {armEnd, pendulumEnd};
}

// REMAP for arcs
function buildThetaArc(theta){
    const pts = [];
    for(let i=0;i<50;i++){
        const t = theta * i / 49;
        pts.push(remap(new THREE.Vector3(
            arcRadius*Math.cos(t),
            arcRadius*Math.sin(t),
            0
        )));
    }
    thetaArc.geometry.dispose();
    thetaArc.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    thetaArc.computeLineDistances();
}

function buildAlphaArc(armEnd, theta, alpha){
    const pts = [];
    for(let i=0;i<50;i++){
        const a = alpha * i / 49;
        pts.push(remap(new THREE.Vector3(
            armEnd.x + arcRadius*Math.sin(a)*Math.sin(theta),
            armEnd.y - arcRadius*Math.sin(a)*Math.cos(theta),
            armEnd.z + arcRadius*Math.cos(a)
        )));
    }
    alphaArc.geometry.dispose();
    alphaArc.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    alphaArc.computeLineDistances();
}

// vertical reference line
updateLine(vertical, remap(armEnd.clone()), remap(armEnd.clone().add(new THREE.Vector3(0,0,1))));