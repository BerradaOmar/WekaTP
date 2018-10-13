import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import java.util.Enumeration;

public class NaivesBayesClassifieur {

    public static void main(String[] args) {

        String file ="C:/Users/BERRADA/IdeaProjects/WekaTP1/res/trainEx2.arff";
        String newFile = "C:/Users/BERRADA/IdeaProjects/WekaTP1/res/testEx2.arff";
        ConverterUtils.DataSource sourceTrain;
        Instances instancesTrain = null;



        try {

            //donner pour l'apprentisage train
            sourceTrain  = new ConverterUtils.DataSource(file);
            instancesTrain = sourceTrain.getDataSet();
            instancesTrain.setClassIndex(instancesTrain.numAttributes()-1);

            //definir es options
            String[] option = {
                     new String("-t"),
                     file,
                    new String("-x"),
                    "4",
                    new String("-i")

            };

            //instancier le classifieur
            NaiveBayes classifieur = new NaiveBayes();
            System.out.println("Constructio du classifieur bayesien naif.");

            // construire le classifier avec données train
            classifieur.buildClassifier(instancesTrain);
            System.out.println("\n Evaluation");

            //evaluer par cross validation
            String result = Evaluation.evaluateModel(classifieur,option);
            System.out.println(result);


            // classement de nouveaux champignons
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(newFile);
            Instances dataset = source.getDataSet();
            dataset.setClassIndex(dataset.numAttributes()-1);
            System.out.println("\n classement de nouveauc champignons ");

            @SuppressWarnings("unchecked")
            Enumeration<Instance> e = dataset.enumerateInstances();
            while (e.hasMoreElements()){
                Instance i = (Instance)e.nextElement();
                double prediction = classifieur.classifyInstance(i);
                System.out.println("Instance <"+i+">\n-> class = "+prediction+" ("+dataset.classAttribute().value((int)prediction)+")");


            }


        }
        catch (Exception e ){

            System.out.println(e);

        }

    }
}
