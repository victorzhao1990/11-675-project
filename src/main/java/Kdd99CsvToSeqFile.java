/**
 * Created by victorzhao on 4/21/15.
 */

import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

// import au.com.bytecode.opencsv.CSVReader;
import com.opencsv.CSVReader;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class Kdd99CsvToSeqFile {

    private String csvPath;
    private Path seqPath;
    private SequenceFile.Writer writer;
    private Configuration conf = new Configuration();
    private Map<String, Long> word2LongMap = Maps.newHashMap();
    private List<String> strLabelList = Lists.newArrayList();
    private FileSystem fs = null;

    public Kdd99CsvToSeqFile(String csvFilePath, String seqPath) {
        this.csvPath = csvFilePath;
        this.seqPath = new Path(seqPath);
    }

    public Map<String, Long> getWordMap() {
        return word2LongMap;
    }

    public List<String> getLabelList() {
        return strLabelList;
    }

    /**
     * Show out the already sequenced file content
     */
    public void dump() {
        try {
            fs = FileSystem.get(conf);
            SequenceFile.Reader reader = new SequenceFile.Reader(fs, this.seqPath, conf);
            Text key = new Text();
            VectorWritable value = new VectorWritable();
            while (reader.next(key, value)) {
                System.out.println( "reading key:" + key.toString() +" with value " +
                        value.toString());
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                fs.close();
                fs = null;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Sequence target csv file.
     * @param labelIndex
     * @param hasHeader
     */
    public void parse(int labelIndex, boolean hasHeader) {
        CSVReader reader = null;
        try {
            fs = FileSystem.getLocal(conf);
            if(fs.exists(this.seqPath))
                fs.delete(this.seqPath, true);
            writer = SequenceFile.createWriter(fs, conf, this.seqPath, Text.class, VectorWritable.class);
            reader = new CSVReader(new FileReader(this.csvPath));
            String[] header = null;
            if(hasHeader) header = reader.readNext();
            String[] line = null;
            Long l = 0L;
            while((line = reader.readNext()) != null) {
                if(labelIndex > line.length) break;
                l++;
                List<String> tmpList = Lists.newArrayList(line);
                String label = tmpList.get(labelIndex);
                if(!strLabelList.contains(label)) strLabelList.add(label);
//				Text key = new Text("/" + label + "/" + l);
                Text key = new Text("/" + label + "/");
                tmpList.remove(labelIndex);

                VectorWritable vectorWritable = new VectorWritable();

                // Just initialize the space for the vector
                Vector vector = new RandomAccessSparseVector(tmpList.size(), tmpList.size());


                for(int i = 0; i < tmpList.size(); i++) {
                    String tmpStr = tmpList.get(i);
                    if(StringUtils.isNumeric(tmpStr))
                        vector.set(i, Double.parseDouble(tmpStr));
                    else
                        vector.set(i, parseStrCell(tmpStr));
                }
                vectorWritable.set(vector);
                writer.append(key, vectorWritable);
            }

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                fs.close();
                fs = null;
                writer.close();
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private Long parseStrCell(String str) {
        Long id = word2LongMap.get(str);
        if( id == null) {
            id = (long) (word2LongMap.size() + 1);
            word2LongMap.put(str, id);
        }
        return id;
    }
}
