export LIST_PS='pgrep -f uvicorn SimpleHTTPServer'
export CLEAR_PS='pkill -f  uvicorn SimpleHTTPServer'

echo 'closing the SimpleHTTPServer and uvicorn processes at'
exec $LIST_PS
exec $CLEAR_PS