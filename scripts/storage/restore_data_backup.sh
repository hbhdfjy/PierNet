#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR=${1:-}
ARCHIVE_NAME=${2:-piern-parquet-20260508.tar.gz}

if [[ -z "$BACKUP_DIR" ]]; then
  echo "usage: bash scripts/storage/restore_data_backup.sh /path/to/data-backup-branch [archive-name]" >&2
  exit 2
fi

if [[ ! -d "$BACKUP_DIR" ]]; then
  echo "backup directory not found: $BACKUP_DIR" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXPORT_DIR="$ROOT/migration_exports"
mkdir -p "$EXPORT_DIR"

PART_GLOB="$BACKUP_DIR/${ARCHIVE_NAME}.part-*"
if ! compgen -G "$PART_GLOB" > /dev/null; then
  echo "no backup chunks matched: $PART_GLOB" >&2
  exit 2
fi

cat $PART_GLOB > "$EXPORT_DIR/$ARCHIVE_NAME"
cp "$BACKUP_DIR/${ARCHIVE_NAME}.sha256" "$EXPORT_DIR/${ARCHIVE_NAME}.sha256"
(
  cd "$ROOT"
  sha256sum -c "migration_exports/${ARCHIVE_NAME}.sha256"
  tar -xzf "migration_exports/${ARCHIVE_NAME}"
)

echo "restored Parquet data into $ROOT/data"
echo "next: python scripts/storage/build_catalog_db.py"
