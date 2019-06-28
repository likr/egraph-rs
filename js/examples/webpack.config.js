const path = require('path')
const CopyWebpackPlugin = require('copy-webpack-plugin')

const options = {
  module: {
    rules: [
      {
        test: /\.js$/,
        include: [path.resolve(__dirname, 'src')],
        use: [
          {
            loader: 'babel-loader',
            options: {
              plugins: ['@babel/plugin-syntax-dynamic-import'],
              presets: ['@babel/preset-env', '@babel/preset-react']
            }
          }
        ]
      }
    ]
  },
  entry: {
    bundle: './src/index.js'
  },
  output: {
    path: path.resolve(__dirname, 'public'),
    filename: '[name].js'
  },
  plugins: [
    new CopyWebpackPlugin([
      {
        from: 'node_modules/bulma/css/bulma.min.css',
        to: 'vendor'
      }
    ])
  ],
  node: {
    crypto: 'empty',
    path: 'empty',
    fs: 'empty'
  },
  devServer: {
    contentBase: path.join(__dirname, 'public'),
    historyApiFallback: true,
    port: 8080
  }
}

if (process.env.NODE_ENV !== 'production') {
  Object.assign(options, {
    devtool: 'inline-source-map'
  })
}

module.exports = options
