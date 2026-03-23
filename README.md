brew install ffmpeg@7
brew link --overwrite ffmpeg@7
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"