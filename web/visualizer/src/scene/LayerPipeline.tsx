import { Text } from "@react-three/drei";
import { useMemo } from "react";
import type { StepSnapshot } from "../inference/snapshot.js";
import { diverging, rgbToCss, sequentialBlue, sequentialOrange } from "./colors.js";
import { HeatmapPlane } from "./HeatmapPlane.js";
import { attentionLastPositionByHead, kvCacheToPositionMajor, lastRow } from "./tensorViews.js";

const STAGE_SPACING = 2.6;
const INK = "#e6e8ee";
const MUTED = "#898781";

// Stable, module-level function references — HeatmapPlane's effect depends on
// these, so passing a fresh arrow function every render would re-run it (and
// churn the update loop) far more than necessary during fast token generation.
const normalizeActivation = (v: number): number => v / 3;
const normalizeMagnitude = (v: number): number => Math.abs(v) / 3;

function mean(data: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i]!;
  return sum / data.length;
}

interface StageBoxProps {
  x: number;
  label: string;
  activation: Float32Array | null;
}

function StageBox({ x, label, activation }: StageBoxProps) {
  const color = activation ? rgbToCss(diverging(mean(activation) / 3)) : "#2a2e3a";
  return (
    <group position={[x, 1.6, 0]}>
      <mesh>
        <boxGeometry args={[1, 0.8, 0.8]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Text position={[0, 0.65, 0]} fontSize={0.18} color={INK} anchorX="center" anchorY="bottom">
        {label}
      </Text>
    </group>
  );
}

interface BlockColumnProps {
  x: number;
  layerIndex: number;
  numHeads: number;
  dK: number;
  contextLength: number;
  step: StepSnapshot | null;
}

function BlockColumn({ x, layerIndex, numHeads, dK, contextLength, step }: BlockColumnProps) {
  const layer = step?.layers[layerIndex];
  const activation = layer ? lastRow(layer.blockOutput) : null;
  const attn = layer ? attentionLastPositionByHead(layer.attnWeights) : null;
  const kCache = step ? kvCacheToPositionMajor(step.kvCacheK[layerIndex] ?? null) : null;
  const vCache = step ? kvCacheToPositionMajor(step.kvCacheV[layerIndex] ?? null) : null;

  return (
    <group>
      <StageBox x={x} label={`block ${layerIndex}`} activation={activation} />

      {/* Activation strip: this block's output for the current position, one cell per model dim. */}
      <HeatmapPlane
        width={activation?.length ?? 1}
        height={1}
        data={activation}
        colormap={diverging}
        normalize={normalizeActivation}
        planeWidth={1.4}
        planeHeight={0.22}
        position={[x, 1.05, 0]}
      />

      {/* Attention: one row per head, columns = key position attended to. */}
      <HeatmapPlane
        width={contextLength}
        height={numHeads}
        data={attn}
        colormap={sequentialBlue}
        planeWidth={1.4}
        planeHeight={0.5}
        position={[x, 0.55, 0]}
      />

      {/* KV cache: rows = position (grows down as generation proceeds), columns = head*dK feature. */}
      <HeatmapPlane
        width={numHeads * dK}
        height={contextLength}
        data={kCache}
        colormap={sequentialBlue}
        normalize={normalizeMagnitude}
        planeWidth={1.4}
        planeHeight={1.3}
        position={[x, -0.35, 0]}
      />
      <HeatmapPlane
        width={numHeads * dK}
        height={contextLength}
        data={vCache}
        colormap={sequentialOrange}
        normalize={normalizeMagnitude}
        planeWidth={1.4}
        planeHeight={1.3}
        position={[x, -1.75, 0]}
      />
    </group>
  );
}

interface LayerPipelineProps {
  step: StepSnapshot | null;
  numLayers: number;
  numHeads: number;
  dK: number;
  contextLength: number;
}

export function LayerPipeline({ step, numLayers, numHeads, dK, contextLength }: LayerPipelineProps) {
  const stages = numLayers + 3; // embedding, N blocks, final norm, lm_head
  const startX = (-(stages - 1) * STAGE_SPACING) / 2;

  const embeddingRow = step ? lastRow(step.embeddingOutput) : null;
  const lnFinalRow = step ? lastRow(step.lnFinalOutput) : null;
  const logitsRow = step ? lastRow(step.logits) : null;

  const lineGeometry = useMemo(() => {
    const points: [number, number, number][] = [];
    for (let i = 0; i < stages; i++) points.push([startX + i * STAGE_SPACING, 1.6, 0]);
    return points;
  }, [stages, startX]);

  return (
    <group>
      {lineGeometry.slice(0, -1).map((p, i) => {
        const next = lineGeometry[i + 1]!;
        const mid: [number, number, number] = [(p[0] + next[0]) / 2, p[1], p[2]];
        const length = next[0] - p[0];
        return (
          <mesh key={i} position={mid} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.02, 0.02, length, 8]} />
            <meshStandardMaterial color={MUTED} />
          </mesh>
        );
      })}

      <StageBox x={startX} label="embed" activation={embeddingRow} />

      {Array.from({ length: numLayers }, (_, i) => (
        <BlockColumn
          key={i}
          x={startX + (i + 1) * STAGE_SPACING}
          layerIndex={i}
          numHeads={numHeads}
          dK={dK}
          contextLength={contextLength}
          step={step}
        />
      ))}

      <StageBox x={startX + (numLayers + 1) * STAGE_SPACING} label="ln_final" activation={lnFinalRow} />
      <StageBox x={startX + (numLayers + 2) * STAGE_SPACING} label="lm_head" activation={logitsRow} />

      {step && (
        <Text
          position={[startX + (numLayers + 2) * STAGE_SPACING, 2.5, 0]}
          fontSize={0.22}
          color={INK}
          anchorX="center"
        >
          → "{step.tokenText}"
        </Text>
      )}

      <Text position={[startX - 1.2, -0.35, 0]} fontSize={0.14} color={MUTED} anchorX="right">
        K: position ↓ · head·d_k →
      </Text>
      <Text position={[startX - 1.2, -1.75, 0]} fontSize={0.14} color={MUTED} anchorX="right">
        V: position ↓ · head·d_k →
      </Text>
      <Text position={[startX - 1.2, 0.55, 0]} fontSize={0.14} color={MUTED} anchorX="right">
        attn: head ↓ · key position →
      </Text>
    </group>
  );
}
