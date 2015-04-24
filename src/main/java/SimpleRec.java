/**
 * Created by victorzhao on 4/19/15.
 */

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import java.io.File;
import java.util.List;

public class SimpleRec {
    public static void main(String[] args) throws Exception {

        // LOADING DATA (RATINGS) FROM FILE
//        final String dir = System.getProperty("user.dir");
//        System.out.println("current dir = " + dir);

        DataModel model = new FileDataModel(new File("./src/main/resources/ml-100k/ua.base"));

        // USER PEARSON CORRELATION FOR COMPUTING SIMILARITY
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

        // LIMITING THE AREA OF USERS FOR COMPARING TO
        UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.5, similarity, model);

        // SENDING EVERYTHING TO RECOMMENDATION ENGINE
        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

        // RECOMMENDING 5 ITEMS FOR FIRST USER (ID: 1)
        List<RecommendedItem> recommendations = recommender.recommend(2, 5);

        // PRESENTING RECOMMENDATIONS
        for (RecommendedItem recommendation : recommendations) {
            System.out.println(recommendation);
        }
    }
}

