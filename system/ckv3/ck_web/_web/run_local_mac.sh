#

# use these to run it locally without docker

brew install node

# sudo apt-get install npm
# --
#package.json:
#{
#    "name": "playwright-express-app",
#    "version": "1.0.0",
#    "description": "A simple Express server to navigate and interact with web pages using Playwright.",
#    "main": "server.js",
#    "scripts": {
#      "start": "node server.js"
#    },
#    "keywords": [
#      "express",
#      "playwright",
#      "automation"
#    ],
#    "author": "",
#    "license": "ISC",
#    "dependencies": {
#      "express": "^4.17.1",
#      "playwright": "^1.28.1"
#    }
#}
# --
# Change to the directory containing package.json
cd "$(dirname "$0")"

# Install dependencies
npm install

# Install fnm (Fast Node Manager)
curl -fsSL https://fnm.vercel.app/install | bash

# activate fnm
source ~/.zshrc

# Verify npm version
npm -v

# Install Playwright and its dependencies
npx playwright install
npx playwright install-deps

# Start the server
npm start
