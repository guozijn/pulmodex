import React, { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const RISK_COLORS = {
  high: 0xff725c,
  mod: 0xffc247,
  low: 0x5cd6a0,
};

const SELECTED_COLOR = 0x6dd3ff;

function riskLevel(prob) {
  if (prob >= 0.7) return "high";
  if (prob >= 0.4) return "mod";
  return "low";
}

function candidateProb(candidate) {
  return typeof candidate.fp_prob === "number" ? candidate.fp_prob : (candidate.prob ?? 0);
}

function buildBounds(candidates) {
  const box = new THREE.Box3();
  candidates.forEach((candidate) => {
    const radius = Math.max(5, (candidate.diameter_mm ?? 8) * 0.75);
    box.expandByPoint(new THREE.Vector3(
      candidate.coordX - radius,
      candidate.coordY - radius,
      candidate.coordZ - radius,
    ));
    box.expandByPoint(new THREE.Vector3(
      candidate.coordX + radius,
      candidate.coordY + radius,
      candidate.coordZ + radius,
    ));
  });
  if (box.isEmpty()) {
    box.setFromCenterAndSize(new THREE.Vector3(0, 0, 0), new THREE.Vector3(140, 140, 140));
  }
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  size.x = Math.max(size.x + 80, 140);
  size.y = Math.max(size.y + 80, 140);
  size.z = Math.max(size.z + 80, 140);
  return { center, size };
}

function planeMaterial(color, opacity) {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
}

export default function NoduleViewer3D({ candidates = [], selectedNodule = null, onSelect }) {
  const mountRef = useRef(null);
  const summary = useMemo(() => {
    if (!candidates.length) {
      return { total: 0, maxDiameter: null, focus: "Awaiting detections" };
    }
    const maxDiameter = Math.max(...candidates.map((candidate) => candidate.diameter_mm ?? 0));
    return {
      total: candidates.length,
      maxDiameter,
      focus: selectedNodule ? "Focused finding" : "Overview",
    };
  }, [candidates, selectedNodule]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    const width = mount.clientWidth || 320;
    const height = mount.clientHeight || 320;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1016);
    scene.fog = new THREE.Fog(0x0b1016, 220, 900);

    const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 5000);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      return undefined;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xd5e6ff, 0.45);
    scene.add(ambient);

    const hemi = new THREE.HemisphereLight(0xe7f2ff, 0x091019, 0.85);
    scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
    keyLight.position.set(1.8, 2.2, 2.4);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x77b9ff, 0.4);
    rimLight.position.set(-1.5, -1.2, 1.8);
    scene.add(rimLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 90;
    controls.maxDistance = 1400;
    controls.zoomSpeed = 0.8;
    controls.panSpeed = 0.7;

    const clickableMeshes = [];
    const { center, size } = buildBounds(candidates);
    const focus = selectedNodule
      ? new THREE.Vector3(selectedNodule.coordX, selectedNodule.coordY, selectedNodule.coordZ)
      : center.clone();

    controls.target.copy(focus);

    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.set(
      focus.x + maxDim * 0.92,
      focus.y - maxDim * 0.78,
      focus.z + maxDim * 0.86,
    );
    camera.near = Math.max(0.1, maxDim / 500);
    camera.far = maxDim * 30;
    camera.updateProjectionMatrix();

    const volumeShell = new THREE.Mesh(
      new THREE.BoxGeometry(size.x, size.y, size.z),
      new THREE.MeshPhongMaterial({
        color: 0x15202b,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    volumeShell.position.copy(center);
    scene.add(volumeShell);

    const boxEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z)),
      new THREE.LineBasicMaterial({ color: 0x5f7488, transparent: true, opacity: 0.42 }),
    );
    boxEdges.position.copy(center);
    scene.add(boxEdges);

    const planeWidth = Math.max(size.x, size.y, size.z) * 1.04;
    const planeHeight = planeWidth;

    const axialPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(planeWidth, planeHeight),
      planeMaterial(0x2f9fc5, 0.08),
    );
    axialPlane.position.copy(focus);

    const coronalPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(planeWidth, planeHeight),
      planeMaterial(0x4e7fbe, 0.06),
    );
    coronalPlane.rotation.x = Math.PI / 2;
    coronalPlane.position.copy(focus);

    const sagittalPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(planeWidth, planeHeight),
      planeMaterial(0x71917f, 0.06),
    );
    sagittalPlane.rotation.y = Math.PI / 2;
    sagittalPlane.position.copy(focus);
    scene.add(axialPlane);
    scene.add(coronalPlane);
    scene.add(sagittalPlane);

    const axisLength = maxDim * 0.72;
    const axisOffset = new THREE.Vector3(
      center.x - size.x / 2 + axisLength * 0.22,
      center.y - size.y / 2 + axisLength * 0.22,
      center.z - size.z / 2 + axisLength * 0.22,
    );

    const axes = [
      { dir: new THREE.Vector3(axisLength, 0, 0), color: 0x7dd3fc },
      { dir: new THREE.Vector3(0, axisLength, 0), color: 0x94f7b2 },
      { dir: new THREE.Vector3(0, 0, axisLength), color: 0xffb676 },
    ];

    axes.forEach(({ dir, color }) => {
      const points = [axisOffset.clone(), axisOffset.clone().add(dir)];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 });
      scene.add(new THREE.Line(geometry, material));
    });

    candidates.forEach((candidate, index) => {
      const prob = candidateProb(candidate);
      const level = riskLevel(prob);
      const isSelected =
        selectedNodule != null &&
        selectedNodule.coordX === candidate.coordX &&
        selectedNodule.coordY === candidate.coordY &&
        selectedNodule.coordZ === candidate.coordZ;

      const radius = Math.max(4.5, (candidate.diameter_mm ?? 8) * 0.48);
      const color = isSelected ? SELECTED_COLOR : RISK_COLORS[level];
      const position = new THREE.Vector3(candidate.coordX, candidate.coordY, candidate.coordZ);

      const core = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 28, 22),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: isSelected ? 0.35 : 0.18,
          roughness: 0.38,
          metalness: 0.08,
          transparent: true,
          opacity: isSelected ? 1 : 0.92,
        }),
      );
      core.position.copy(position);
      core.userData.candidateIndex = index;
      scene.add(core);
      clickableMeshes.push(core);

      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(radius * (isSelected ? 1.9 : 1.55), 20, 16),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: isSelected ? 0.14 : 0.08,
          side: THREE.BackSide,
          depthWrite: false,
        }),
      );
      halo.position.copy(position);
      scene.add(halo);

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius * 1.55, Math.max(0.5, radius * 0.08), 12, 40),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: isSelected ? 0.92 : 0.4,
        }),
      );
      ring.position.copy(position);
      ring.rotation.x = Math.PI / 2.5;
      scene.add(ring);
    });

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    function handleClick(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(clickableMeshes, false);
      if (hits.length > 0) {
        const hitIndex = hits[0].object.userData.candidateIndex;
        onSelect?.(candidates[hitIndex]);
      }
    }

    renderer.domElement.addEventListener("click", handleClick);

    const resizeObserver = new ResizeObserver(() => {
      const nextWidth = mount.clientWidth;
      const nextHeight = mount.clientHeight;
      if (!nextWidth || !nextHeight) return;
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    });
    resizeObserver.observe(mount);

    let animationFrameId = 0;
    function animate() {
      animationFrameId = window.requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    return () => {
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("click", handleClick);
      controls.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose?.();
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach((material) => material.dispose?.());
          } else {
            object.material.dispose?.();
          }
        }
      });
      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [candidates, onSelect, selectedNodule]);

  return (
    <div
      ref={mountRef}
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        background: "linear-gradient(180deg, #111a23 0%, #0b1016 58%, #070b10 100%)",
      }}
    >
      <div style={{
        position: "absolute",
        inset: 10,
        border: "1px solid rgba(146, 168, 189, 0.16)",
        boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.02)",
        pointerEvents: "none",
        zIndex: 1,
      }} />

      <div style={{
        position: "absolute",
        top: 12,
        left: 12,
        display: "flex",
        flexDirection: "column",
        gap: 3,
        pointerEvents: "none",
        zIndex: 2,
      }}>
        <div style={{
          fontSize: 9,
          fontFamily: "var(--mono)",
          color: "rgba(173, 206, 235, 0.62)",
          letterSpacing: "0.14em",
          fontWeight: 600,
        }}>
          3D VOLUME
        </div>
        <div style={{
          fontSize: 10,
          color: "rgba(255,255,255,0.78)",
          letterSpacing: "0.02em",
        }}>
          {summary.focus}
        </div>
      </div>

      <div style={{
        position: "absolute",
        top: 12,
        right: 12,
        display: "grid",
        gap: 4,
        pointerEvents: "none",
        zIndex: 2,
        textAlign: "right",
      }}>
        <div style={{ fontSize: 9, fontFamily: "var(--mono)", color: "rgba(255,255,255,0.42)" }}>
          {summary.total} finding{summary.total === 1 ? "" : "s"}
        </div>
        {summary.maxDiameter != null && (
          <div style={{ fontSize: 9, fontFamily: "var(--mono)", color: "rgba(255,255,255,0.42)" }}>
            MAX {summary.maxDiameter.toFixed(1)} MM
          </div>
        )}
      </div>

      <div style={{
        position: "absolute",
        left: 18,
        bottom: 18,
        display: "grid",
        gap: 4,
        pointerEvents: "none",
        zIndex: 2,
      }}>
        <div style={{ fontSize: 9, fontFamily: "var(--mono)", color: "#7dd3fc", letterSpacing: "0.1em" }}>R/L</div>
        <div style={{ fontSize: 9, fontFamily: "var(--mono)", color: "#94f7b2", letterSpacing: "0.1em" }}>A/P</div>
        <div style={{ fontSize: 9, fontFamily: "var(--mono)", color: "#ffb676", letterSpacing: "0.1em" }}>S/I</div>
      </div>

      {candidates.length > 0 && (
        <div style={{
          position: "absolute",
          bottom: 12,
          right: 12,
          fontSize: 9,
          fontFamily: "var(--mono)",
          color: "rgba(255,255,255,0.3)",
          letterSpacing: "0.05em",
          pointerEvents: "none",
          zIndex: 2,
          textAlign: "right",
          lineHeight: 1.5,
        }}>
          left drag orbit
          <br />
          wheel zoom
          <br />
          click marker focus
        </div>
      )}

      {candidates.length === 0 && (
        <div style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 11,
          color: "rgba(255,255,255,0.24)",
          letterSpacing: "0.02em",
          pointerEvents: "none",
          zIndex: 2,
        }}>
          No nodules to display
        </div>
      )}
    </div>
  );
}
