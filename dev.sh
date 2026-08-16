#!/bin/zsh
export PATH="$HOME/.nvm/versions/node/v24.18.0/bin:$PATH"
cd "$(dirname "$0")"
exec npx vite --port 5183
