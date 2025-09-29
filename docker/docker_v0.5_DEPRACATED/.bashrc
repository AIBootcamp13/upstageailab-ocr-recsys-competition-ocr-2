# === OCR Development Environment .bashrc ===
# A rich shell experience with a modern prompt, robust environment management, and utility functions.

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# Correctly unset stale VIRTUAL_ENV if it's invalid or points to an outside project
if [ -n "$VIRTUAL_ENV" ]; then
    # Check if the directory doesn't exist OR if it's not inside the current path
    if [ ! -d "$VIRTUAL_ENV" ] || [[ "$VIRTUAL_ENV" != "$PWD"* ]]; then
        unset VIRTUAL_ENV
    fi
fi

# === History Configuration ===
HISTCONTROL=ignoreboth:ignoredups:ignorespace
HISTSIZE=10000
HISTFILESIZE=20000
shopt -s histappend
set +H

# === Locale Configuration ===
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8
if locale -a | grep -q "ko_KR.utf8"; then
    export LC_MESSAGES=ko_KR.UTF-8
    export LC_TIME=ko_KR.UTF-8
    export LANGUAGE=en_US:ko_KR
fi

# === Color Prompt Configuration ===
# A modern, concise prompt showing venv, current directory, and git branch.
if [ "$TERM" != "dumb" ]; then
    # Define colors
    CYAN='\[\033[0;36m\]'
    YELLOW='\[\033[0;33m\]'
    GREEN='\[\033[0;32m\]'
    BLUE='\[\033[0;34m\]'
    RESET='\[\033[0m\]'

    # Function to get the active virtual environment name
    parse_venv_name() {
        if [ -n "$VIRTUAL_ENV" ]; then
            printf "(%s)" "${VIRTUAL_ENV##*/}"
        fi
    }

    # Function to get the current git branch
    parse_git_branch() {
        git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
    }

    # Set the final prompt string (PS1)
    PS1="${CYAN}\$(parse_venv_name)${BLUE} \W ${YELLOW}\$(parse_git_branch)${RESET}${GREEN} ❯ ${RESET}"
fi

# === Python Environment Management (PyEnv, UV, Conda) ===
# Manually add pyenv's shims and bin to the PATH for predictable behavior
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# === UV Environment Management ===
# UV is the primary package manager for this project
export UV_LINK_MODE=copy
export PATH="$HOME/.cargo/bin:$PATH"

# Auto-activate UV environment if .venv exists in current directory
uv_auto_activate() {
    if [ -f ".venv/bin/activate" ] && [ -z "$VIRTUAL_ENV" ]; then
        source .venv/bin/activate
        export UV_ACTIVE=1
    fi
}

# Check for UV environment on directory change
PROMPT_COMMAND="${PROMPT_COMMAND:+$PROMPT_COMMAND$'\n'}uv_auto_activate"

# UV aliases for common commands
alias uv-run='uv run'
alias uv-add='uv add'
alias uv-sync='uv sync --group dev'
alias uv-test='uv run pytest tests/ -v --tb=short'
alias uv-lint='uv run flake8 .'
alias uv-format='uv run black . && uv run isort .'

# Properly initialize Conda to override any external activation scripts.
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# After initializing, ensure unwanted features are turned off.
if command -v conda >/dev/null 2>&1; then
    conda config --set auto_activate_base false
    conda config --set changeps1 false
fi

# === Aliases ===
# Enable color support for common commands
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# Common navigation and file aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gd='git diff'
alias gco='git checkout'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up -d'
alias dcd='docker-compose down'
alias dcl='docker-compose logs'
alias dce='docker-compose exec'
alias dhelper="./docker/docker-helper.sh"

# OCR Project aliases (using the robust python -m pytest pattern)
alias train='uv run python runners/train.py'
alias predict='uv run python runners/predict.py'
alias run-tests='uv run python -m pytest tests/'

# === Utility Functions ===
# Extract various archive formats
extract() {
    if [ -f $1 ]; then
        case $1 in
            *.tar.bz2)   tar xjf $1     ;; *.tar.gz)    tar xzf $1     ;;
            *.bz2)       bunzip2 $1     ;; *.rar)       unrar e $1     ;;
            *.gz)        gunzip $1      ;; *.tar)       tar xf $1      ;;
            *.tbz2)      tar xjf $1     ;; *.tgz)       tar xzf $1     ;;
            *.zip)       unzip $1       ;; *.Z)         uncompress $1  ;;
            *.7z)        7z x $1        ;; *)           echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# Create and enter directory
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# === Final PATH & Environment Setup ===
# Add local user bin to PATH
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi

# Development environment variables
export EDITOR=vim

# Enable programmable completion features
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi
