import { TransformerLM, Tokenizer, stateDictFromBinary, type WeightsManifest } from "@spiderman/nn";

export interface LoadedModel {
  model: TransformerLM;
  tokenizer: Tokenizer;
  manifest: WeightsManifest;
}

/**
 * Fetches manifest.json + weights.bin + vocab.json + merges.json from
 * `${baseUrl}weights/` (copied there at build/dev time by
 * scripts/copy-weights.mjs) and builds a ready-to-use model + tokenizer.
 * `baseUrl` should be `import.meta.env.BASE_URL` — a relative fetch, so this
 * works both in local dev (served from `/`) and on GitHub Pages (served
 * from `/spiderman/`).
 */
export async function loadModel(baseUrl: string): Promise<LoadedModel> {
  const weightsBase = `${baseUrl}weights/`;

  const [manifest, buffer, rawVocab, rawMerges] = await Promise.all([
    fetch(`${weightsBase}manifest.json`).then((r) => r.json() as Promise<WeightsManifest>),
    fetch(`${weightsBase}weights.bin`).then((r) => r.arrayBuffer()),
    fetch(`${weightsBase}vocab.json`).then((r) => r.json()),
    fetch(`${weightsBase}merges.json`).then((r) => r.json()),
  ]);

  const model = new TransformerLM({
    vocabSize: manifest.config.vocabSize,
    contextLength: manifest.config.contextLength,
    dModel: manifest.config.dModel,
    numLayers: manifest.config.numLayers,
    numHeads: manifest.config.numHeads,
    dFF: manifest.config.dFF,
    ropeTheta: manifest.config.ropeTheta,
  });
  model.loadWeights(stateDictFromBinary(manifest, buffer));

  const tokenizer = Tokenizer.fromJSON(rawVocab, rawMerges, manifest.specialTokens);

  return { model, tokenizer, manifest };
}
