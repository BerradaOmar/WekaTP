import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class ClassifJ48 {

    public static void main(String[] args){
       /* if(args.length < 2){
            System.out.println("Usage : J48 <arff_train_file> <arff_test_file>");
            System.exit(1);
        }*/
        String trainFile = "C:/Users/BERRADA/IdeaProjects/WekaTP1/res/trainEx2.arff";
        String testFile = "C:/Users/BERRADA/IdeaProjects/WekaTP1/res/testEx2.arff";
        ConverterUtils.DataSource sourceTrain;
        ConverterUtils.DataSource sourceTest;
        Instances instancesTrain = null;
        Instances instancesTest = null;

        try{
            //donnees pour l'apprentissage
            sourceTrain = new ConverterUtils.DataSource(trainFile);
            instancesTrain = sourceTrain.getDataSet();
            instancesTrain.setClassIndex(instancesTrain.numAttributes()-1);

            //donnees pour l'evaluation
            sourceTest = new ConverterUtils.DataSource(testFile);
            instancesTest = sourceTest.getDataSet();
            instancesTest.setClassIndex (instancesTest.numAttributes()-1);

            //definir les options
            String[] options = {
                    new String("-t"),
                    trainFile,

                    new String("-T"),
                    testFile,

                    //parametres specifiques a J48
                    new String("-C"),
                    new String("0.25"),

                    new String("-M"),
                    new String("2")
            };

            //instancier le classifieur
            J48 classifieur = new J48();

            //construire le classifieur avec donnees train
            classifieur.buildClassifier(instancesTrain);

            //evaluer avec donnees test
            String result = Evaluation.evaluateModel(classifieur,options);
            System.out.println(result);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}



