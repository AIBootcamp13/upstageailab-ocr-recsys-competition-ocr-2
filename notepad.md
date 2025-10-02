alias mcb='omake serve-ui'
alias mev='omake serve-evaluation-ui'
alias minf='omake serve-inference-ui'

function ui-train() { omake serve-ui PORT="${1:-8502}"; }
function ui-eval()  { omake serve-evaluation-ui PORT="${1:-8503}"; }
function ui-infer() { omake serve-inference-ui PORT="${1:-8504}"; }
