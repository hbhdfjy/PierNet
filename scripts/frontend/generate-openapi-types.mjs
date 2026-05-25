import { existsSync, mkdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const thisFile = fileURLToPath(import.meta.url)
const repoRoot = resolve(dirname(thisFile), '..', '..')
const frontendRoot = join(repoRoot, 'frontend')
const schemaPath = join(frontendRoot, 'src', 'lib', 'generated', 'openapi.json')
const typesPath = join(frontendRoot, 'src', 'lib', 'generated', 'openapi.d.ts')
const localPython = join(repoRoot, '.conda', 'env', 'bin', 'python')
const python = process.env.PIERN_PYTHON || (existsSync(localPython) ? localPython : 'python3')
const openapiTypescript = join(frontendRoot, 'node_modules', '.bin', process.platform === 'win32' ? 'openapi-typescript.cmd' : 'openapi-typescript')

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    stdio: 'inherit',
    env: {
      ...process.env,
      PYTHONPATH: process.env.PYTHONPATH ? `${repoRoot}:${process.env.PYTHONPATH}` : repoRoot,
    },
    ...options,
  })
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(' ')} failed with exit code ${result.status}`)
  }
}

mkdirSync(dirname(schemaPath), { recursive: true })
run(python, ['scripts/ci/export_openapi.py', schemaPath])

if (!existsSync(openapiTypescript)) {
  throw new Error('openapi-typescript is not installed; run npm install in frontend first')
}

run(openapiTypescript, [schemaPath, '-o', typesPath])
