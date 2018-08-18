const path = require('path')
const CopyWebpackPlugin = require('copy-webpack-plugin')

const options = {
  module: {
    rules: [
      {
        test: /\.js$/,
        include: [
          path.resolve(__dirname, 'src')
        ],
        use: [
          {
            loader: 'babel-loader',
            options: {
              presets: ['env']
            }
          }
        ]
      }
    ]
  },
  entry: {
    'force-directed': './src/force-directed/index',
    'group-in-a-box': './src/group-in-a-box/index'
  },
  output: {
    path: path.resolve(__dirname, 'public'),
    filename: '[name]/bundle.js'
  },
  plugins: [
    new CopyWebpackPlugin([
      'node_modules/egraph/egraph.wasm'
    ])
  ],
  node: {
    'crypto': 'empty',
    'path': 'empty',
    'fs': 'empty'
  },
  devServer: {
    contentBase: path.join(__dirname, 'public'),
    historyApiFallback: true,
    port: 8084
  }
}

if (process.env.NODE_ENV !== 'production') {
  Object.assign(options, {
    devtool: 'inline-source-map'
  })
}

module.exports = options
