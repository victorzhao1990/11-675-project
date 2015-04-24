import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

//import testbayes.CsvToVectors;
//import testbayes.MahoutVector;
import com.opencsv.CSVReader;

public class MainTest {

	//private static String trainFile = "./src/main/resources/kddcup99/kddcup.data_10_percent_corrected";
	private static String trainFile = "./src/main/resources/kddcup99/kddcup.data.corrected";
	private static String testFile = "./src/main/resources/kddcup99/corrected";
	private static String trainSeqFile = "./src/main/resources/kddcup99-seq/train-10-pct-seq-2";
	private static String testSeqFile = "./src/main/resources/kddcup99-seq/corrected-seq";
	private static NaiveBayesModel naiveBayesModel = null;
	private static Map<String, Long> strOptionMap = null;
	private static List<String> strLabelList = null;
	public static void main(String[] args) {
		
		try {
			// Step 1 : Convert CSV to Sequence file
			Kdd99CsvToSeqFile trainingCsvtoSeq = new Kdd99CsvToSeqFile(trainFile, trainSeqFile);
			trainingCsvtoSeq.parse(41, false);
			strOptionMap = trainingCsvtoSeq.getWordMap();
			strLabelList = trainingCsvtoSeq.getLabelList();
			
			// Step 2: Train NB
			train();
			
			// Step 3: Test to see result
			test();
		} catch (Throwable e) {
			e.printStackTrace();
		}
		
		
	}
	
	public static void train() throws Throwable
	{

		System.out.println("~~~ begin to train ~~~");
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.getLocal(conf);
		TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
		trainNaiveBayes.setConf(conf);
		
		String outputDirectory = "./src/main/resources/kddcup99-bayes/output";
		String tempDirectory = "./src/main/resources/kddcup99-bayes/temp";
		
		fs.delete(new Path(outputDirectory),true);
		fs.delete(new Path(tempDirectory),true);
		// cmd sample: mahout trainnb -i train-vectors -el -li labelindex -o model -ow -c
		trainNaiveBayes.run(new String[] { 
				"--input", trainSeqFile, 
				"--output", outputDirectory,
				"-el", 
				"--labelIndex", "labelIndex",
				"--overwrite", 
				"--tempDir", tempDirectory });
		
		// Train the classifier
		naiveBayesModel = NaiveBayesModel.materialize(new Path(outputDirectory), conf);

		System.out.println("features: " + naiveBayesModel.numFeatures());
		System.out.println("labels: " + naiveBayesModel.numLabels());
	}
	
	public static void test() throws IOException {
		System.out.println("~~~ begin to test ~~~");
	    AbstractVectorClassifier classifier = new ComplementaryNaiveBayesClassifier(naiveBayesModel);
	    
	    CSVReader csv = new CSVReader(new FileReader(testFile));
	    csv.readNext(); // skip header
	    String[] line = null;
	    double totalSampleCount = 0.;
	    double correctClsCount = 0.;
	    while((line = csv.readNext()) != null) {
	    	totalSampleCount ++;
	    	Vector vector = new RandomAccessSparseVector(40,40);//???
	    	for(int i = 0; i < 40; i++) {
	    		if(StringUtils.isNumeric(line[i])) {
	    			vector.set(i, Double.parseDouble(line[i]));
	    		} else {
	    			Long id = strOptionMap.get(line[i]);
	    			if(id != null)
	    				vector.set(i, id);
	    			else {
	    				System.out.println(StringUtils.join(line, ","));
	    				continue;
	    			}
	    		}
	    	}
    		Vector resultVector = classifier.classifyFull(vector);
			int classifyResult = resultVector.maxValueIndex();
			if(StringUtils.equals(line[41], strLabelList.get(classifyResult))) {
		    	correctClsCount++;
		    } else {
		    	System.out.println("Correct=" + line[41] + "\tClassify=" + strLabelList.get(classifyResult) );
		    }
		    
	    }
	    
	    System.out.println("Correct Ratio:" + (correctClsCount / totalSampleCount));
	    
	    
	}
	

}
