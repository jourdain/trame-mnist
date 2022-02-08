const path = require('path');
const DST_PATH = '../trame_ai_lenet_5/html/module/serve';

module.exports = {
  outputDir: path.resolve(__dirname, DST_PATH),
  configureWebpack: {
    output: {
      libraryExport: 'default',
    },
  },
  transpileDependencies: [],
};
