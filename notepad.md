## Streamlit CLI aliases
alias mcb='omake serve-ui'
alias mev='omake serve-evaluation-ui'
alias minf='omake serve-inference-ui'

function ui-train() { omake serve-ui PORT="${1:-8502}"; }
function ui-eval()  { omake serve-evaluation-ui PORT="${1:-8503}"; }
function ui-infer() { omake serve-inference-ui PORT="${1:-8504}"; }


## Resource Monitoring
To monitor for resource leaks in the future, you can run:
ps aux | grep -E '(claude|mcp-)' | grep -v grep | wc -l


# Human-Readable Bytes
ps -eo pid,user,%cpu,%mem,rss:20,vsz,tty,stat,start,time,cmd --sort -rss
