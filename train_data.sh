#Linux retraining command
WORKING_DIR="tf_files"

BOTTLENECK_DIR="$WORKING_DIR/bottlenecks"
STEPS=3000
MODEL_DIR="$WORKING_DIR/inception"
OUTPUT_GRAPH="$WORKING_DIR/retrained_graph.pb"
OUTPUT_LABELS="$WORKING_DIR/retrained_labels.txt"
DATA_FOLDER="$WORKING_DIR/data"
SUM_FOLDER="$WORKING_DIR/retrain_logs"

python train.py \
--bottleneck_dir=$BOTTLENECK_DIR  \
--how_many_training_steps $STEPS \
--saved_model_dir=$MODEL_DIR \
--output_graph=$OUTPUT_GRAPH \
--output_labels=$OUTPUT_LABELS  \
--summaries_dir=$SUM_FOLDER \
--image_dir=$DATA_FOLDER

#Windows retraining command
#python retrain.py --bottleneck_dir=bottlenecks --how_many_training_steps 500 --model_dir=model_dir --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --summaries_dir=retrain_logs --image_dir=images
