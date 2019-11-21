package cmps142_hw4;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.function.IntToDoubleFunction;
import java.lang.*;
import java.util.*;

public class LogisticRegression {

        /** the learning rate */
        private double rate=0.01;

        /** the weights to learn */
        private double[] weights;

        /** the number of iterations */
        private int ITERATIONS = 200;

        /** TODO: Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. **/
        public LogisticRegression(int n) { // n is the number of weights to be learned
            this.weights = new double[n];
        }

        /** TODO: Implement the function that returns the L2 norm of the weight vector **/
        // L2 norm and magnitude are equivalent wording.
        // This function calcs the sqrt(every element ^ 2 together)
        private double weightsL2Norm(){
            double sum = 0.0;
            for(double w : this.weights){
                sum = sum + w * w;
            }
            return Math.sqrt(sum);
        }

        /** TODO: Implement the sigmoid function **/
        private static double sigmoid(double z) {
            return 1.0 / (Math.exp(-1 * z) + 1);
        }

        /** TODO: Helper function for prediction **/
        /** Takes a test instance as input and outputs the probability of the label being 1 **/
        /** This function should call sigmoid() **/
        private double probPred1(double[] x) {
            // Since sigmoid returns a value between 0-1.
            // this returns the probability of our input vector (x[]) dotted
            // with weight vector (this.weight[])
            return sigmoid(dotProd(x));
        }

        // returns the dot product of input vector x and our global weight vector.
        private double dotProd(double[] x){
          double sum = 0.0;
          for(int i = 0; i < x.length; i++){
              sum = sum + x[i] * this.weights[i];
          }
          return sum;
        }

        /** TODO: The prediction function **/
        /** Takes a test instance as input and outputs the predicted label **/
        /** This function should call probPred1() **/
        public int predict(double[] x) {
          // if the probability returns higher than .5, then w*x is classified as positive.
            if(probPred1(x) >= (double) 0.5){
                return 1;
            }
            else {
                return 0;
            }
        }

        /** This function takes a test set as input, call the predict() to predict a label for it, and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix **/
        public void printPerformance(List<LRInstance> testInstances) {
            double acc = 0;
            double p_pos = 0, r_pos = 0, f_pos = 0;
            double p_neg = 0, r_neg = 0, f_neg = 0;
            int TP=0, TN=0, FP=0, FN=0; // TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives

            // TODO: write code here to compute the above mentioned variables
            for(int i = 0; i < testInstances.size(); i++){
                int actualValue = testInstances.get(i).label;
                int prediction = predict(testInstances.get(i).x);
                if(prediction == 1){
                    if(prediction == actualValue){
                        TP += 1;
                    }
                    else {
                        FP += 1;
                    }
                }
                else if(prediction == 0){
                    if(prediction == actualValue){
                        TN += 1;
                    }
                    else{
                        FN += 1;
                    }
                }
            }
            acc = (double)(TP + TN) / (testInstances.size());
            p_pos = (double) TP / (TP + FP);
            p_neg = (double) TN / (TN + FN);
            r_pos = (double) TP / (TP + FN);
            r_neg = (double) TN / (TN + FP);
            f_pos = (2 * p_pos * r_pos) / (p_pos + r_pos);
            f_neg = (2 * p_neg * r_neg) / (p_neg + r_neg);

            System.out.println("Accuracy="+acc);
            System.out.println("P, R, and F1 score of the positive class=" + p_pos + " " + r_pos + " " + f_pos);
            System.out.println("P, R, and F1 score of the negative class=" + p_neg + " " + r_neg + " " + f_neg);
            System.out.println("Confusion Matrix");
            System.out.println(TP + "\t" + FN);
            System.out.println(FP + "\t" + TN);
        }


        /** Train the Logistic Regression using Stochastic Gradient Ascent **/
        /** Also compute the log-likelihood of the data in this function **/
        public void train(List<LRInstance> instances) {
            for (int n = 0; n < ITERATIONS; n++) {
                double lik = 0.0; // Stores log-likelihood of the training data for this iteration
                for (int i=0; i < instances.size(); i++) {
                    // TODO: Train the model
                    double [] X = instances.get(i).x;
                    int Y = instances.get(i).label;
                    double prob = probPred1(X);

                    for(int j = 0; j < this.weights.length; j++){
                        this.weights[j] = this.weights[j] + this.rate * X[j] * (Y - prob);

                    }
                    // TODO: Compute the log-likelihood of the data here. Remember to take logs when necessary
                    // this says: for correct label Y, multiply times log of prob x is labeled correctly (1),
                    // minus incorrect label times log of probability of X being labeled incorrectly (0).
                    // as interpreted from:
                    // https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf

                    //lik = Y * (Math.log(prob)) + (1 - Y) * (Math.log(1-prob));
                    // This is from slides 5-1 #24. l(W)=SUM_l(Y^l * {SUM_0^d(w_i*X_i^l)} - log(1+EXP(SUM_0^d(w_i * X_i^l))))
                    // 2nd for-loop = SUM_l()
                    // Y is the ith label in our instance. (Y^l)
                    // dotProd = vector X dotted with Weight vector (SUM(w_i * X_i^l))
                    // prob = probability of X being 1. (EXP(SUM_0^d(w_i * X-i^l)))
                    // lik = lik + (Y * Math.log(prob) + (1 - Y) * Math.log(1 - prob));
                    double product = dotProd(X);
                    lik = lik + (Y * product - Math.log(1 + Math.exp(product)));
                }
                System.out.println("iteration: " + n + " lik: " + lik);
            }
            // System.out.println("final weight vector: " + Arrays.toString(this.weights));
        }

        public static class LRInstance {
            public int label; // Label of the instance. Can be 0 or 1
            public double[] x; // The feature vector for the instance

            /** TODO: Constructor for initializing the Instance object **/
            public LRInstance(int label, double[] x) {
                this.label = label;
                this.x = x;
            }
        }

        /** Function to read the input dataset **/
        public static List<LRInstance> readDataSet(String file) throws FileNotFoundException {
            List<LRInstance> dataset = new ArrayList<LRInstance>();
            Scanner scanner = null;
            try {
                scanner = new Scanner(new File(file));

                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    if (line.startsWith("ju")) { // Ignore the header line
                        continue;
                    }
                    String[] columns = line.replace("\n", "").split(",");

                    // every line in the input file represents an instance-label pair
                    int i = 0;
                    double[] data = new double[columns.length - 1];
                    for (i=0; i < columns.length - 1; i++) {
                        data[i] = Double.valueOf(columns[i]);
                    }
                    int label = Integer.parseInt(columns[i]); // last column is the label
                    LRInstance instance = new LRInstance(label, data); // create the instance
                    dataset.add(instance); // add instance to the corpus
                }
            } finally {
                if (scanner != null)
                    scanner.close();
            }
            return dataset;
        }


        public static void main(String... args) throws FileNotFoundException {
            List<LRInstance> trainInstances = readDataSet("HW4_trainset.csv");
            List<LRInstance> testInstances = readDataSet("HW4_testset.csv");

            // create an instance of the classifier
            int d = trainInstances.get(0).x.length;
            LogisticRegression logistic = new LogisticRegression(d);

            logistic.train(trainInstances);

            System.out.println("Norm of the learned weights = "+logistic.weightsL2Norm());
            System.out.println("Length of the weight vector = "+logistic.weights.length);

            // printing accuracy for different values of lambda
            System.out.println("-----------------Printing train set performance-----------------");
            logistic.printPerformance(trainInstances);

            System.out.println("-----------------Printing test set performance-----------------");
            logistic.printPerformance(testInstances);
        }

    }
