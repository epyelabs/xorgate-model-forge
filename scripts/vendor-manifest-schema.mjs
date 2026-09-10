#!/usr/bin/env node
// Vendor the engine's own model manifest JSON Schema (zod -> JSON Schema, as
// the engine's catalog does) into contracts/model-manifest.schema.json, so
// the forge validates manifests against the schema the engine defines, not a
// retyped copy. Run `node scripts/vendor-manifest-schema.mjs [out]`; the
// golden suite re-emits and diffs.
import { writeFileSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { modelManifestSchema } = require('@xorgate/workflow-engine');
const { zodToJsonSchema } = require('zod-to-json-schema');
const engineVersion = require('@xorgate/workflow-engine/package.json').version;

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = process.argv[2] ?? join(root, 'contracts', 'model-manifest.schema.json');
const schema = zodToJsonSchema(modelManifestSchema, { name: 'ModelManifest', $refStrategy: 'none' });
const doc = {
  $schema: schema.$schema ?? 'http://json-schema.org/draft-07/schema#',
  $id: 'https://docs.xorgate.io/schemas/forge/model-manifest.schema.json',
  title: `model manifest v1 (vendored from @xorgate/workflow-engine ${engineVersion}, src/models/registry.ts)`,
  'x-engineVersion': engineVersion,
  ...schema.definitions.ModelManifest,
};
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(doc, null, 2) + '\n');
console.log(out);
