import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { LoadingState } from "./components/Feedback";

const BuildPage = lazy(() =>
  import("./pages/BuildPage").then((module) => ({ default: module.BuildPage })),
);
const CreateProjectPage = lazy(() =>
  import("./pages/CreateProjectPage").then((module) => ({
    default: module.CreateProjectPage,
  })),
);
const DemoPage = lazy(() =>
  import("./pages/DemoPage").then((module) => ({ default: module.DemoPage })),
);
const ProjectOverviewPage = lazy(() =>
  import("./pages/ProjectOverviewPage").then((module) => ({
    default: module.ProjectOverviewPage,
  })),
);
const ProjectsPage = lazy(() =>
  import("./pages/ProjectsPage").then((module) => ({
    default: module.ProjectsPage,
  })),
);
const ResourcesPage = lazy(() =>
  import("./pages/ResourcesPage").then((module) => ({
    default: module.ResourcesPage,
  })),
);

export function App() {
  return (
    <Suspense fallback={<LoadingState label="正在打开页面" />}>
      <Routes>
        <Route path="/" element={<ProjectsPage />} />
        <Route path="/new" element={<CreateProjectPage />} />
        <Route path="/projects/:projectId" element={<ProjectOverviewPage />} />
        <Route path="/projects/:projectId/resources" element={<ResourcesPage />} />
        <Route path="/projects/:projectId/mapping" element={<ResourcesPage />} />
        <Route path="/projects/:projectId/build" element={<BuildPage />} />
        <Route path="/projects/:projectId/demo" element={<DemoPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}
