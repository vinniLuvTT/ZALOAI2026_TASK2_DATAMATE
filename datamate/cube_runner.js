// Programmatic Cube Core server. Config comes from env (see dm/cube_server.py).
// cwd must be the per-business folder that contains ./model/*.yml
const CubejsServer = require('@cubejs-backend/server');

const server = new CubejsServer({ schemaPath: 'model' });
server.listen().then(({ port }) => {
  console.log(`CUBE_READY ${port}`);
}).catch((e) => {
  console.error('CUBE_FAILED', e);
  process.exit(1);
});
