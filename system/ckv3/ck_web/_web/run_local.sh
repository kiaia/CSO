#

# use these to run it locally without docker

sudo apt-get install npm
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
npm install
# --
# update node.js according to "https://nodejs.org/en/download/package-manager"
# installs fnm (Fast Node Manager)
curl -fsSL https://fnm.vercel.app/install | bash

# activate fnm
source ~/.bashrc

# download and install Node.js
fnm use --install-if-missing 22

# verifies the right Node.js version is in the environment
node -v # should print `v22.11.0`

# verifies the right npm version is in the environment
npm -v # should print `10.9.0`
# --
npx playwright install
npx playwright install-deps
npm install uuid
npm install js-yaml
npm install playwright-extra puppeteer-extra-plugin-stealth
npm install async-mutex

# --
# simply run it with

npm start
