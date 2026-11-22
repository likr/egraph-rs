import { useEffect, useRef } from "react";

export function drawGraphToCanvas(canvas, scene) {
  if (!canvas || !scene || !scene.coords || scene.numNodes === 0) {
    return;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 800;
  const height = canvas.clientHeight || 560;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const { coords, numNodes, links = [], nodes = [], bundled, nodeRadius = 4.5, showLabels = false } = scene;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < numNodes; i++) {
    const x = coords[2 * i];
    const y = coords[2 * i + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) {
    ctx.restore();
    return;
  }

  const pad = 36;
  const spanX = Math.max(maxX - minX, 1e-4);
  const spanY = Math.max(maxY - minY, 1e-4);
  const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY);
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const toScreenX = (x) => width * 0.5 + (x - cx) * scale;
  const toScreenY = (y) => height * 0.5 + (y - cy) * scale;

  ctx.strokeStyle = "rgba(120, 130, 145, 0.35)";
  ctx.lineWidth = numNodes > 300 ? 0.6 : 1.1;

  if (bundled && bundled.offsets && bundled.points) {
    const { offsets, points } = bundled;
    const nEdges = Math.max(0, offsets.length - 1);
    for (let e = 0; e < nEdges; e++) {
      const start = offsets[e];
      const end = offsets[e + 1];
      if (end - start < 2) continue;
      ctx.beginPath();
      ctx.moveTo(toScreenX(points[2 * start]), toScreenY(points[2 * start + 1]));
      for (let p = start + 1; p < end; p++) {
        ctx.lineTo(toScreenX(points[2 * p]), toScreenY(points[2 * p + 1]));
      }
      ctx.stroke();
    }
  } else {
    ctx.beginPath();
    for (let i = 0; i < links.length; i++) {
      const { source, target } = links[i];
      ctx.moveTo(toScreenX(coords[2 * source]), toScreenY(coords[2 * source + 1]));
      ctx.lineTo(toScreenX(coords[2 * target]), toScreenY(coords[2 * target + 1]));
    }
    ctx.stroke();
  }

  const r = numNodes > 300 ? Math.min(nodeRadius, 2.5) : nodeRadius;
  for (let i = 0; i < numNodes; i++) {
    const sx = toScreenX(coords[2 * i]);
    const sy = toScreenY(coords[2 * i + 1]);
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = nodes[i]?.color ?? "#3273dc";
    ctx.fill();
    if (numNodes <= 200) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.0;
      ctx.stroke();
    }
    if (showLabels && nodes[i]?.name) {
      ctx.fillStyle = "#222222";
      ctx.font = "10px sans-serif";
      ctx.fillText(nodes[i].name, sx + r + 2, sy + 3);
    }
  }

  ctx.restore();
}

export function GraphCanvas({ sceneRef, tick }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    drawGraphToCanvas(canvasRef.current, sceneRef.current);
  }, [tick, sceneRef]);

  useEffect(() => {
    const onResize = () => drawGraphToCanvas(canvasRef.current, sceneRef.current);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [sceneRef]);

  return (
    <div className="graph-canvas-wrap">
      <canvas ref={canvasRef} data-testid="graph-canvas" />
    </div>
  );
}
