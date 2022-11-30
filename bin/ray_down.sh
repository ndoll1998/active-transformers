
SESSION=cluster

# create a tmux session
tmux has-session -t $SESSION 2>/dev/null
# create tmux session
if [ $? != 0 ]; then
    echo "WARNING: session not found"
    exit
fi


for WINDOW in $(tmux list-windows -t $SESSION -F '#{window_name}'); do
    echo "Stopping node $WINDOW"

    tmux send-keys -t "$SESSION:$WINDOW" C-c
    tmux send-keys -t "$SESSION:$WINDOW" "ray stop" C-m
done

# wait some time and kill session
sleep 5
tmux kill-session -t "$SESSION"
