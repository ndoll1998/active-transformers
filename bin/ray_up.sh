
HEAD="dpl13"
PORT=8080
NODES="dpl11 dpl15 dpl16 dpl17"

SESSION=cluster
# create a tmux session
tmux has-session -t $SESSION 2>/dev/null
# create tmux session
if [ $? != 0 ]; then
    echo "Creating Session: $SESSION"
    tmux new-session -s "$SESSION" -d
fi

# start head node
echo "Starting head node..."
tmux new-window -t "$SESSION" -n "head"
tmux send-keys -t "$SESSION:head" "ssh $HEAD" C-m
tmux send-keys -t "$SESSION:head" "conda activate al" C-m
tmux send-keys -t "$SESSION:head" "ray start --head --port $PORT" C-m
tmux send-keys -t "$SESSION:head" "watch ray status" C-m

# start worker nodes
for NODE in $NODES; do

    echo starting worker node $NODE
    tmux new-window -t "$SESSION" -n "$NODE"
    tmux send-keys -t "$SESSION:$NODE" "ssh $NODE" C-m
    tmux send-keys -t "$SESSION:$NODE" "conda activate al" C-m
    tmux send-keys -t "$SESSION:$NODE" "ray start --address $HEAD:$PORT" C-m

done

exit
NODES="dpl01 dpl02 dpl03 dpl04 dpl05"
# start worker nodes
for NODE in $NODES; do

    echo starting worker node $NODE
    tmux new-window -t "$SESSION" -n "$NODE"
    tmux send-keys -t "$SESSION:$NODE" "ssh $NODE" C-m
    tmux send-keys -t "$SESSION:$NODE" "conda activate al" C-m
    tmux send-keys -t "$SESSION:$NODE" "ray start --address $HEAD:$PORT --num-gpus 3" C-m

done
