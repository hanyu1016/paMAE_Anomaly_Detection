import tensorflow as tf
# import mlflow

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        # self.writer = tf.summary.FileWriter(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)
        

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            # self.writer.add_summary(summary, step)
            self.writer.flush()
        
        
        # mlflow.log_metric(tag, value, step=step)