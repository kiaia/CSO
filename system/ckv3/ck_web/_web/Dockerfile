FROM node:latest as build

#WORKDIR /app

RUN useradd -rm -d /home/zzz -s /bin/bash -g root -G sudo -u 1001 zzz
USER zzz
WORKDIR /home/zzz

COPY package*.json /home/zzz/
RUN mkdir -p ./DownloadedFiles ./screenshots
COPY . .

RUN npm install

RUN npx playwright install
USER root
RUN bash -c "echo root:pwd | chpasswd"
RUN npx playwright install-deps
RUN chown -R zzz ./
USER zzz

RUN npm install uuid
RUN npm install js-yaml
RUN npm install playwright-extra puppeteer-extra-plugin-stealth

CMD ["npm", "start"]


# cp -r $PYTHONPATH/mspx/tasks/ck/system/proj_ck/_web .
# cd _web
# docker build . -t myweb
# docker run -it -p 8833:3000 myweb bash
# npm start
# --
# adduser --disabled-password --gecos "" zzz
# chown -R zzz ./
# su zzz
# --
# wget https://raw.githubusercontent.com/microsoft/playwright/c0d9ccfebf023d04a033cddcacc4a5fd47d520e0/docs/docker/seccomp_profile.json
# docker run --rm -it --ipc=host --security-opt seccomp=seccomp_profile.json -p 8833:3000 myweb bash
# while true; do timeout -k 1m -s 9 1h npm start; done
# timeout -k 1m -s 9 30m npm start; while true; do timeout -k 1m -s 9 1h npm start; done
# --
