# PiERN Studio Frontend

独立于旧 `frontend` 的用户工作台。源码、依赖、路由、样式、测试和构建产物均在本目录内。

```bash
npm install
npm run dev
```

- Studio 开发端口：`3001`
- API：`/api/studio/*`
- Vite 默认代理：`http://127.0.0.1:8000`
- 生产路径：`/studio`

可通过 `PIERN_API_TARGET` 覆盖开发代理目标。
