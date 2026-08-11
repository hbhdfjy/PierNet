import clsx from "clsx";
import { CheckCircle2, FileArchive, FileSpreadsheet, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

import { formatBytes } from "../lib/format";
import type { DataResource, ExpertResource } from "../types";
import { Button } from "./Button";

interface UploadSurfaceProps {
  kind: "data" | "expert";
  resource: DataResource | ExpertResource | null;
  busy: boolean;
  onUpload: (file: File) => Promise<void>;
}

const copy = {
  data: {
    title: "科学计算数据",
    detail: "成对的输入与输出样本",
    accept: ".npz,.h5,.hdf5,.csv,.parquet,.zip,.tar.gz,.tgz",
    formats: "科学数据文件、表格或压缩包",
    icon: FileSpreadsheet,
  },
  expert: {
    title: "计算模型",
    detail: "接收输入并返回数值结果",
    accept: ".py,.zip,.tar.gz,.tgz",
    formats: "Python 文件或模型压缩包",
    icon: FileArchive,
  },
};

export function UploadSurface({ kind, resource, busy, onUpload }: UploadSurfaceProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const item = copy[kind];
  const Icon = item.icon;

  const choose = (file?: File) => {
    if (file) void onUpload(file);
  };

  return (
    <section className={clsx("upload-section", resource && "upload-section--complete")}>
      <div className="upload-section__heading">
        <span className="upload-section__icon">
          <Icon size={20} />
        </span>
        <div>
          <h2>{item.title}</h2>
          <p>{item.detail}</p>
        </div>
        {resource ? <CheckCircle2 className="upload-section__check" size={21} /> : null}
      </div>

      {resource ? (
        <div className="uploaded-file">
          <div>
            <strong>{resource.filename}</strong>
            <span>
              {formatBytes(resource.size_bytes)}
              {" · "}
              {kind === "data"
                ? (resource as DataResource).samples
                  ? `${(resource as DataResource).samples} 条样本`
                  : "等待识别"
                : `${(resource as ExpertResource).file_count} 个文件`}
            </span>
          </div>
          <Button variant="quiet" onClick={() => inputRef.current?.click()} disabled={busy}>
            更换
          </Button>
        </div>
      ) : (
        <button
          type="button"
          className={clsx("dropzone", dragging && "dropzone--dragging")}
          onClick={() => inputRef.current?.click()}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            choose(event.dataTransfer.files[0]);
          }}
          disabled={busy}
        >
          <UploadCloud size={25} aria-hidden="true" />
          <strong>{busy ? "正在上传" : "选择文件或拖到这里"}</strong>
          <span>{item.formats}</span>
        </button>
      )}

      <input
        ref={inputRef}
        className="sr-only"
        type="file"
        accept={item.accept}
        onChange={(event) => choose(event.target.files?.[0])}
        tabIndex={-1}
      />
    </section>
  );
}
