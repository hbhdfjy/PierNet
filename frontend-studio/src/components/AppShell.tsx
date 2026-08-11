import { ArrowLeft, FolderKanban } from "lucide-react";
import type { PropsWithChildren } from "react";
import { Link, useLocation } from "react-router-dom";

import type { ProjectSnapshot } from "../types";
import { StageRail } from "./StageRail";

function Brand() {
  return (
    <Link className="brand" to="/">
      <span className="brand__mark" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      <span className="brand__wordmark">
        <strong>PiERN</strong>
        <small>Studio</small>
      </span>
    </Link>
  );
}

export function AppShell({
  project,
  children,
}: PropsWithChildren<{ project?: ProjectSnapshot | null }>) {
  const location = useLocation();
  const insideProject = location.pathname.includes("/projects/");

  return (
    <div className="app-shell">
      <header className="topbar">
        <Brand />
        <nav className="topbar__nav" aria-label="主导航">
          {insideProject ? (
            <Link to="/" className="topbar__link">
              <ArrowLeft size={16} />
              所有项目
            </Link>
          ) : (
            <Link to="/" className="topbar__link">
              <FolderKanban size={16} />
              项目
            </Link>
          )}
        </nav>
      </header>
      <div className={project ? "shell-body shell-body--project" : "shell-body"}>
        {project ? <StageRail project={project} /> : null}
        <main className="page">{children}</main>
      </div>
    </div>
  );
}
