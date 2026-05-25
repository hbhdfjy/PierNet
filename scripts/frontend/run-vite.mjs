#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const MIN_NODE_VERSION = [20, 19, 0]
const MIN_NODE_LABEL = '20.19.0'
const REEXEC_ENV = 'PierNet_VITE_NODE_REEXEC'

const scriptPath = fileURLToPath(import.meta.url)
const projectRoot = resolve(dirname(scriptPath), '../..')
const viteBin = join(projectRoot, 'frontend', 'node_modules', 'vite', 'bin', 'vite.js')
const args = process.argv.slice(2)

function nodeVersionParts(versionText) {
  const match = versionText.trim().match(/^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?/)
  return match ? [Number(match[1]), Number(match[2] ?? 0), Number(match[3] ?? 0)] : [0, 0, 0]
}

function nodeMeetsMinimum(versionText) {
  const current = nodeVersionParts(versionText)
  for (let i = 0; i < MIN_NODE_VERSION.length; i += 1) {
    if (current[i] > MIN_NODE_VERSION[i]) return true
    if (current[i] < MIN_NODE_VERSION[i]) return false
  }
  return true
}

function candidateNodes() {
  const seen = new Set()
  const nodes = []
  const add = (candidate) => {
    if (!candidate || seen.has(candidate)) return
    seen.add(candidate)
    nodes.push(candidate)
  }

  add(process.env.PierNet_NODE_BIN)
  add(process.env.PierNet_NODE)
  add(join(projectRoot, '.node', 'current', 'bin', 'node'))
  const localNodeRoot = join(projectRoot, '.node')
  if (existsSync(localNodeRoot)) {
    for (const entry of readdirSync(localNodeRoot).sort()) {
      if (entry.startsWith('node-v')) add(join(localNodeRoot, entry, 'bin', 'node'))
    }
  }
  if (process.env.CONDA_PREFIX) add(join(process.env.CONDA_PREFIX, 'bin', 'node'))
  if (process.env.HOME) {
    add(join(process.env.HOME, '.conda', 'envs', 'PierNet', 'bin', 'node'))
    add(join(process.env.HOME, 'miniconda3', 'envs', 'PierNet', 'bin', 'node'))
    add(join(process.env.HOME, 'anaconda3', 'envs', 'PierNet', 'bin', 'node'))
  }
  return nodes
}

function compatibleNode(candidate) {
  if (!existsSync(candidate)) return false
  const result = spawnSync(candidate, ['--version'], { encoding: 'utf8' })
  if (result.status !== 0) return false
  return nodeMeetsMinimum(result.stdout)
}

function reexecWithCompatibleNode() {
  for (const candidate of candidateNodes()) {
    if (!compatibleNode(candidate)) continue
    const result = spawnSync(candidate, [scriptPath, ...args], {
      stdio: 'inherit',
      env: {
        ...process.env,
        [REEXEC_ENV]: '1',
        PATH: `${dirname(candidate)}:${process.env.PATH ?? ''}`,
      },
    })
    if (result.error) {
      console.error(result.error.message)
      process.exit(1)
    }
    process.exit(result.status ?? 1)
  }

  console.error(
    `Vite 需要 Node ${MIN_NODE_LABEL}+，当前是 ${process.version}。` +
    ' 请设置 PierNet_NODE_BIN=/path/to/node，或激活包含新版 Node 的 PierNet conda 环境。',
  )
  process.exit(1)
}

if (!nodeMeetsMinimum(process.version)) {
  if (process.env[REEXEC_ENV]) {
    console.error(`Vite 需要 Node ${MIN_NODE_LABEL}+，当前是 ${process.version}。`)
    process.exit(1)
  }
  reexecWithCompatibleNode()
}

if (!globalThis.crypto?.getRandomValues) {
  const { webcrypto } = await import('node:crypto')
  globalThis.crypto = webcrypto
}

await import(pathToFileURL(viteBin).href)
