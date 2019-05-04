package Jowil;

import Jowil.Reports.Report;
import Jowil.Reports.Report2;
import javafx.util.Pair;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.rank.Percentile ;




import org.apache.commons.math3.distribution.TDistribution ;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import sun.rmi.server.InactiveGroupException;


import java.text.DecimalFormat;
import java.text.Format;
import java.util.*;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import static org.apache.commons.math3.stat.StatUtils.* ;


public class Statistics {



    ////////////////////fields
//    public static final int IDMODE=0,NAMEMODE=1,AUTOMODE=2;
//    private static int identifierMode=AUTOMODE;
    private static ArrayList<String> studentNames;

    private static ArrayList<Double> studentScores;
    private static ArrayList<Integer> studentForms; //zero based form number for each student
    private static ArrayList<ArrayList<Double>> questionWeights; //form vs question weights
    private static ArrayList<String> questionNames;
    private static ArrayList<ArrayList<String>> correctAnswers; //form vs correct answers
    private static ArrayList<ArrayList<String>> studentAnswers; //student vs answers
    private static ArrayList<ArrayList<ArrayList<String>>>sortedStudentAnswers; //for each form :student vs (answers+score)
    private static ArrayList<ArrayList<ArrayList<Double>>>answersStats; //For each form :Questions vs each possible choice percentage ( every row can have different number of possible choices)
    private static ArrayList<ArrayList<Double>> correctAnswersPercents ; // for each form for each question in that form the percentage of studests who got the correct answer
    private static ArrayList<ArrayList<String>> questionsChoices; //list of all possible choices in order for every question. (every row can have different number of choices)
    private static ArrayList<ArrayList<Double>> subjScores; //subjective scores -> student  vs sub scores
    private static ArrayList<String> grades ; // list of university grades i.e. A , B , C
    private static ArrayList<Double> gradesLowerRange; // list of the lower range of the grades assuming that the upper range is the lower range of the next grade
    private static ArrayList<String> studentIdentifier;
    private static String identifierName ="ID";
    private static double maxScore = -1 ;
    private static double bonus = 0 ;
    private static double userMaxScore = -1 ;
    private static boolean allowExceedMaxScore = true ;
    private static double epslon = 0.0000001 ;
    private static boolean addBonusToAll;
    private static boolean isIdentifierNumeric;
    private static ArrayList<Double> subjMaxScores;
    private static ArrayList<ArrayList<Double>> formsScors ;
    private static ArrayList<ArrayList<ArrayList<Integer>>> birnaryStudentResponses; // for each form for each question for each student 1 if he got the correct answer else 0
    private static ArrayList<ArrayList<Integer>> studentsCorrectAnswersCount ; // for each form for each student number of questions he got right

    ////////////////////setters
//    public static void setIdentifierMode(int identifierMode) {
//        Statistics.identifierMode = identifierMode;
//    }
    public static void setStudentNames(ArrayList<String> studentNames) {
        Statistics.studentNames = studentNames;
    }


    public static void setStudentForms(ArrayList<Integer> studentForms) {
        Statistics.studentForms = studentForms;
    }

    public static void setQuestionWeights(ArrayList<ArrayList<Double>> questionWeights) {
        Statistics.questionWeights = questionWeights;
    }

    public static void setQuestionNames(ArrayList<String> questionNames) {
        Statistics.questionNames = questionNames;
    }


    public static void setCorrectAnswers(ArrayList<ArrayList<String>> correctAnswers) {
        Statistics.correctAnswers = correctAnswers;
    }

    public static void setStudentAnswers(ArrayList<ArrayList<String>> studentAnswers) {
        Statistics.studentAnswers = studentAnswers;
    }

    public static void setQuestionsChoices(ArrayList<ArrayList<String>> questionsChoices) {
        Statistics.questionsChoices = questionsChoices;
    }

    public static void setGradesLowerRange(ArrayList<Double> gradesLowerRange) {
        Statistics.gradesLowerRange = gradesLowerRange;
    }

    public static void setGrades(ArrayList<String> grades) {
        Statistics.grades = grades;
    }
    public static void setSubjScores(ArrayList<ArrayList<Double>> subScores) {
        Statistics.subjScores = subScores;
    }


    public static void setStudentIdentifier(ArrayList<String> studentIdentifier) {
        Statistics.studentIdentifier = studentIdentifier;
    }

    public static void setIdentifierName(String idnetifierName) {
        Statistics.identifierName = idnetifierName;
    }

    public static ArrayList<Double> getSubjMaxScores() {
        return subjMaxScores;
    }

    public static void setSubjMaxScores(ArrayList<Double> subjScores) {
        Statistics.subjMaxScores = subjScores;
    }


    public static void setFormsScors(ArrayList<ArrayList<Double>> formsScors) {
        Statistics.formsScors = formsScors;
    }


    public static void setBonus(double bonus) {
        Statistics.bonus = bonus;
    }

    public static void setUserMaxScore(double userMaxScore) {
        Statistics.userMaxScore = userMaxScore;
    }

    public static void setAllowExceedMaxScore(boolean allowExceedMaxScore) {
        Statistics.allowExceedMaxScore = allowExceedMaxScore;
    }

    public static void setAddBonusToAll(boolean addBonusToAll) {
        Statistics.addBonusToAll = addBonusToAll;
    }


    //getters
//    public static int getIdentifierMode() {
//        return identifierMode;
//    }
    public static ArrayList<String> getStudentNames() {
        return studentNames;
    }



    public static ArrayList<Integer> getStudentForms() {
        return studentForms;
    }
    public static ArrayList<String> getQuestionNames() {
        return questionNames;
    }

    public static ArrayList<ArrayList<String>> getCorrectAnswers() {
        return correctAnswers;
    }

    public static ArrayList<ArrayList<String>> getStudentAnswers() {
        return studentAnswers;
    }


    public static ArrayList<ArrayList<Double>> getSubjScores() {
        return subjScores;
    }

    public static ArrayList<String> getStudentIdentifier() {
        return studentIdentifier;
    }

    public static String getIdentifierName() {
        return identifierName;
    }

    public static ArrayList<String> getSpecificQuestionChoices(int questionIndex){return questionsChoices.get(questionIndex);}

    public static int getNumberOfForms (){
        double[]forms = studentForms.stream().mapToDouble(d -> d).toArray();
        return (int)max(forms)+1 ;
    }

    public static boolean isAllowExceedMaxScore() {
        return allowExceedMaxScore;
    }


    public static boolean isAddBonusToAll() {
        return addBonusToAll;
    }

    public static ArrayList<ArrayList<String>> getQuestionsChoices(){
        return questionsChoices;
    }
    public static ArrayList<ArrayList<Double>> getQuestionWeights() {
        return questionWeights;
    }

    public static int getNumberOfStudents (){
        return studentScores.size() ;
    }

    public static boolean isIsIdentifierNumeric() {
        return isIdentifierNumeric;
    }

    public static void setIsIdentifierNumeric(boolean isIdentifierNumeric) {
        Statistics.isIdentifierNumeric = isIdentifierNumeric;
    }


    public static Double getPassingPercent() {
        int gradeIndex =1 ;
        while (gradesLowerRange.get(gradeIndex)*100 < 50)
            gradeIndex++;
        return  gradesLowerRange.get(gradeIndex) ;
    }

    public static Double getMaxScore(){ return maxScore ; }
    public static double getBonus() {
        return bonus;
    }

    public static ArrayList<ArrayList<Double>> getCorrectAnswersPercents() {
        return correctAnswersPercents;
    }
    public static double getUserMaxScore() {
        return userMaxScore;
    }


    public static ArrayList<String> getGrades() {
        return grades;
    }

    // print fuctions
    public static void printStudentScores() {
        System.out.print("Student Scores: ");
        System.out.println(studentScores);
    }
    public static void printSortedStudentAnswers(){
        System.out.print("Sorted Student Answers: ");
        System.out.println(sortedStudentAnswers);
    }

    public static void printAnswerStats(){
        System.out.print("Answer Stats: ");
        System.out.println(answersStats);
    }
    public static void printBasicInfo(){
        System.out.println("-----------------------------------------------------");
        System.out.println("Info Headers: "+CSVHandler.getDetectedInfoHeaders());
        System.out.println("Q names: " + Jowil.Statistics.getQuestionNames().toString());
//        System.out.println("Student Identifiers: " + Jowil.Statistics.getStudentIdentifier().toString());
        //System.out.println("Student names: " + Jowil.Statistics.getStudentNames().toString());
//        System.out.println("ID mode " + Jowil.Statistics.getIdentifierMode());
        System.out.println("Correct ans: " + Jowil.Statistics.getCorrectAnswers().toString());
        System.out.println("Student ans: " + Jowil.Statistics.getStudentAnswers().toString());
        System.out.println("Questions choices: "+ Statistics.questionsChoices);
        System.out.println("Grades: "+grades);
        System.out.println("Grades Lower Ranges: "+gradesLowerRange);
    }
    public static void printCalculations(){
        System.out.println("-----------------------------------------------------");
        printStudentScores();
        printSortedStudentAnswers();
        printAnswerStats();
    }


    private static void printMap (Map<String , String> map ) {
        System.out.println("Map Data");
        for(Map.Entry<String , String> entry : map.entrySet()) {
            System.out.println(entry.getKey() + " : " + entry.getValue());
        }
    }

    private static void printTable (ArrayList<ArrayList<String>> table) {
        System.out.println("Table Data: ");
        for(ArrayList<String> row : table) {
            System.out.println(row);
        }

    }



    ///////////////initializers

    static void initScores(){

        initMaxScore();
        studentScores = new ArrayList<Double>() ;
        for(int i = 0 ; i <studentAnswers.size() ; i ++) {
            double studentScore = calcStudentScore(i) ;
            studentScores.add(studentScore) ;
        }
        printStudentScores();
    }

    public static void initFormsScores(){
        if(maxScore ==-1)
            initMaxScore();

        double[]forms = studentForms.stream().mapToDouble(d -> d).toArray();
        int numberOfForms = (int)max(forms)+1 ;

        formsScors = new ArrayList<ArrayList<Double>>() ;

        for (int formIndex =0 ; formIndex < numberOfForms ; formIndex++) {
            ArrayList<Double> formScores = new ArrayList<>() ;
            formsScors.add(formScores);
        }
        for(int i = 0 ; i <studentAnswers.size() ; i ++) {
            double studentScore = calcStudentScore(i) ;
            formsScors.get(studentForms.get(i)).add(studentScore) ;
        }
//        printStudentScores();
        for (int formIndex =0 ; formIndex < numberOfForms ; formIndex++) {
            System.out.println("form "+(formIndex+1)+" answers: "+formsScors.get(formIndex)) ;
        }
    }



    private static void initSortedStudentAnswers(){

        for(int i=0;i<studentAnswers.size();i++)
            studentAnswers.get(i).add(studentScores.get(i).toString());

        sortedStudentAnswers=new ArrayList<ArrayList<ArrayList<String>>>();
        for(int i=0;i<correctAnswers.size();i++)
            sortedStudentAnswers.add(new ArrayList<ArrayList<String>>());

        for(int i=0;i<studentAnswers.size();i++)
            sortedStudentAnswers.get(studentForms.get(i)).add(studentAnswers.get(i));

        SortByScore sorter =new SortByScore();
        for(int i=0;i<sortedStudentAnswers.size();i++)
            Collections.sort(sortedStudentAnswers.get(i),sorter);
    }

    public static void initAnswersStats(){

        answersStats=new ArrayList<ArrayList<ArrayList<Double>>>(correctAnswers.size());
        for(int i=0;i<correctAnswers.size();i++){
            answersStats.add(new ArrayList<ArrayList<Double>>());
            calcformAnswerStats(answersStats.get(i),i);

        }

    }

    public static void initMaxScore() {
        if(userMaxScore != -1) {
            maxScore = userMaxScore;
            return;
        }
        double[] wieghts = questionWeights.get(0).stream().mapToDouble(d -> d).toArray();
        maxScore = sum(wieghts) ;
        if(subjMaxScores.size()!=0) {
            double[] subj = subjMaxScores.stream().mapToDouble(d -> d).toArray();
            maxScore += sum(subj) ;
        }

    }

    public static void initCorrectAnswersPercent(){
        correctAnswersPercents = new ArrayList<>( );
        for(int formIndex = 0  ; formIndex < getNumberOfForms()  ; formIndex++){
            ArrayList<ArrayList<Double>> formAnswerStats=answersStats.get(formIndex);

            ArrayList<Double> formCorrectPercents = new ArrayList<>();
            for(int questionIndex = 0  ; questionIndex< formAnswerStats.size() ; questionIndex++) {
                ArrayList<String> formCorrectAnswers = correctAnswers.get(formIndex);
                    String correctAnswer  = formCorrectAnswers.get(questionIndex);
                    int correctAnswerIndex = questionsChoices.get(questionIndex).indexOf(correctAnswer);
                    formCorrectPercents.add(formAnswerStats.get(questionIndex).get(correctAnswerIndex));
                }
            correctAnswersPercents.add(formCorrectPercents) ;
        }
    }


    public static void init(){
        questionNames=CSVHandler.getDetectedQHeaders();
        initScores();
        initFormsScores();
        initPointBiserialTables();
        initSortedStudentAnswers();
        initAnswersStats();
        initCorrectAnswersPercent();
    }

//    public static String formatNumber ( DecimalFormat  fom , double number) {
//
//    return "";
//    }



    //helper functions

    public static int getMaxNumOfChoices(){
        int max = 0  ;
        for (ArrayList<String> choices: questionsChoices) {
            int lenght = choices.size() ;
            if(lenght > max)
                max = lenght ;
        }
        return max ;
    }
    private static double calcStudentScore(int studentIndex){
        double studentScore =0;
        ArrayList<String> studentAnswer = studentAnswers.get(studentIndex) ;
        for (int j = 0 ; j <  studentAnswer.size() ; j ++) {
            String questionAnswer = studentAnswer.get(j) ;
            if(questionAnswer.equals(correctAnswers.get(studentForms.get(studentIndex)).get(j))) {
                studentScore+= questionWeights.get(studentForms.get(studentIndex)).get(j);
            }
        }
        if(subjScores.size()!=0) {
            ArrayList<Double> studentSubjScores = subjScores.get(studentIndex);
            Double subjScore = sum(studentSubjScores.stream().mapToDouble(d -> d).toArray());
            studentScore+=subjScore ;
        }

        //if the student grade was below passing grade or if we are adding bonus to all students
        if((double)studentScore/maxScore<getPassingPercent() || addBonusToAll)
            studentScore+=bonus ;
        if(!allowExceedMaxScore)
            studentScore = studentScore>maxScore?maxScore:studentScore ;
        return studentScore;
    }
    private static void calcformAnswerStats(ArrayList<ArrayList<Double>> formAnswerStats , int formIndex){

        double count;
        int studentsCount=getFormStudentsCount(formIndex);
        for(int i=0;i<questionsChoices.size();i++){
            formAnswerStats.add(new ArrayList<Double>());
            for(String choice: questionsChoices.get(i)){
                count=0;
                for(int j=0;j<studentAnswers.size();j++){
                    if(studentAnswers.get(j).get(i).equals(choice) && studentForms.get(j)==formIndex)
                        count++;
                }
                formAnswerStats.get(i).add(count/studentsCount);
            }
        }

    }
    private static int getFormStudentsCount(int formIndex){
        int count=0;
        for(int form: studentForms){
            if(form==formIndex)
                count++;
        }
        return count;
    }




    private static double calcCI(int sampleSize, double level , double StandardDiviation ) {
        try {
            // Create T Distribution with N-1 degrees of freedom
            TDistribution tDist = new TDistribution(sampleSize - 1 );
            // Calculate critical value
            double critVal = tDist.inverseCumulativeProbability(1.0 - (1 - level) / 2);
            // Calculate confidence interval
            return critVal * StandardDiviation / Math.sqrt(sampleSize);
        } catch (MathIllegalArgumentException e) {
            return Double.NaN;
        }
    }


    private static double calcSEM ( double scoreStd ) {
        return 0  ;
    }

    private static double calcKr20 ( int formIndex){

        ArrayList<Integer> formCAC = studentsCorrectAnswersCount.get(formIndex) ;

        double[] formCACArray = formCAC.stream().mapToDouble(d -> d).toArray();

        double var = variance(formCACArray) ;
        var += epslon ;

        double pqsum = 0 ;
//            double var = studentsCorrectAnswersCount

        ArrayList<ArrayList<Double>> formStats = answersStats.get(formIndex);
        for(int qIndex = 0 ; qIndex < formStats.size() ; qIndex++) {
            ArrayList<Double> questionStats = formStats.get(qIndex);
            String correctAnswer = correctAnswers.get(formIndex).get(qIndex);
            //get index of the correct answer in the question choices List
            int correctAnswerIndex = questionsChoices.get(qIndex).indexOf(correctAnswer);
            // p is the proportion of correct responses
            double p = questionStats.get(correctAnswerIndex);
            // q is the proportion of incorrect responses
            double q = (1 - p);

            pqsum += p * q ;

        }

        int k  = questionsChoices.size();
        if(k == 1)
            return 1 ;
        return k/(k-1) * (1-pqsum/var) ;
    }



    public static double calcKr21 (int formIndex) {

        double n = birnaryStudentResponses.get(0).size() ;
        double[] formCAC = studentsCorrectAnswersCount.get(formIndex).stream().mapToDouble(d -> d).toArray();

        double mean = mean(formCAC) ;
        double var = variance(formCAC) ;

        return n / (n - 1) * (1 - (mean * (n - mean) / (n * var)));
    }

    public static double calcCronbachAlpha (int formIndex) {

        double k = birnaryStudentResponses.get(0).size() ;
        double[] formCAC = studentsCorrectAnswersCount.get(formIndex).stream().mapToDouble(d -> d).toArray();
        double totalVar = variance(formCAC) ;
        double varSum = 0 ;
        for(ArrayList<Integer> questionResponses : birnaryStudentResponses.get(formIndex)) {
           double[] questionRes =  questionResponses.stream().mapToDouble(d -> d).toArray();
            varSum += variance(questionRes) ;
        }

        return  (k/(k-1)) * (1-varSum/totalVar) ;

    }

    public static Map<String , String> calcFormTestInsights(ArrayList<Double> formCorrectPercents) {

        Map <String, String> formTestInsights = new LinkedHashMap<>();
        double groupMinCorrectPercent = 1 ;
        double groupMaxCorrectPercent = 0 ;

        String hardestQuestion = "" ;
        String easiestQuestion = "" ;

        String hardestGroup = "" ;
        String easiestGroup = "" ;

        double[] formPercents = formCorrectPercents.stream().mapToDouble(d -> d).toArray();
        double maxFormPercent = max(formPercents) ;
        double minFormPercent = min(formPercents) ;

        int questionIndex = formCorrectPercents.indexOf(maxFormPercent) ;
        easiestQuestion = questionNames.get(questionIndex);
        questionIndex  = formCorrectPercents.indexOf(minFormPercent) ;
        hardestQuestion = questionNames.get(questionIndex);


        ArrayList<Group> groups = CSVHandler.getDetectedGroups();
        int qIndex=   0 ;
        for(Group group:groups){
            int qCount = group.getqCount() ;
            double[] groupPercents = formCorrectPercents.subList(qIndex , qIndex+qCount).stream().mapToDouble(d -> d).toArray();
            double avgGroupPercent = sum(groupPercents)/qCount ;
            if(avgGroupPercent>groupMaxCorrectPercent){
                groupMaxCorrectPercent = avgGroupPercent ;
                easiestGroup = group.getCleanedName() ;
            }
            if(avgGroupPercent<groupMinCorrectPercent){
                groupMinCorrectPercent =avgGroupPercent ;
                hardestGroup = group.getCleanedName() ;
            }

            qIndex+= qCount ;
        }

        formTestInsights.put("Hardest Question" , hardestQuestion);
        formTestInsights.put("Easiest Question" , easiestQuestion) ;
        formTestInsights.put("Hardest Group" , hardestGroup) ;
        formTestInsights.put("Easiest Group" , easiestGroup) ;

        return formTestInsights ;
    }

    public static Map<String , String> calcTestInsights(){
        Map<String , String> statsMap = new LinkedHashMap<>();
        double minCorrectPercent = 1 ;
        double maxCorrectPercent = 0 ;

        double groupMinCorrectPercent = 1 ;
        double groupMaxCorrectPercent = 0 ;

        String hardestQuestion = "" ;
        String easiestQuestion = "" ;

        String hardestGroup = "" ;
        String easiestGroup = "" ;

        for(int formIndex =0  ; formIndex<getNumberOfForms() ; formIndex++) {
            String form ="" ;

            // string to spicify form that contains the question
            if (getNumberOfForms()>1)
                form = " in Form "+(formIndex+1) ;

            // for hardest and easiest questions
            ArrayList<Double> formCorrectPercents = correctAnswersPercents.get(formIndex);
            double[] formPercents = formCorrectPercents.stream().mapToDouble(d -> d).toArray();
            double maxFormPercent = max(formPercents) ;
            if(maxFormPercent>maxCorrectPercent) {
                maxCorrectPercent = maxFormPercent ;
                int questionIndex = formCorrectPercents.indexOf(maxFormPercent) ;
                easiestQuestion = questionNames.get(questionIndex)+form ;
            }
            double minFormPercent = min(formPercents) ;
            if(minFormPercent<minCorrectPercent) {
                minCorrectPercent = minFormPercent ;
                int questionIndex = formCorrectPercents.indexOf(minFormPercent) ;
                hardestQuestion = questionNames.get(questionIndex)+form ;
            }
            // hardest and easiest groups
            ArrayList<Group> groups = CSVHandler.getDetectedGroups();
            int qIndex=   0 ;
            for(Group group:groups){
                int qCount = group.getqCount() ;
                double[] groupPercents = formCorrectPercents.subList(qIndex , qIndex+qCount).stream().mapToDouble(d -> d).toArray();
                double avgGroupPercent = sum(groupPercents)/qCount ;
                if(avgGroupPercent>groupMaxCorrectPercent){
                    groupMaxCorrectPercent = avgGroupPercent ;
                    easiestGroup = group.getCleanedName() + form ;
                }
                if(avgGroupPercent<groupMinCorrectPercent){
                    groupMinCorrectPercent =avgGroupPercent ;
                    hardestGroup = group.getCleanedName()+form  ;
                }

                qIndex+= qCount ;
            }
        }

        statsMap.put("Hardest Question" , hardestQuestion) ;
        statsMap.put("Easiest Question" , easiestQuestion) ;
        statsMap.put("Hardest Group" , hardestGroup) ;
        statsMap.put("Easiest Group" , easiestGroup) ;
        return  statsMap;
    }

    public static Map<String,String> calcGeneralStats (ArrayList<Double> studentScores ,  int numberOfQuestions , int formIndex) {
        Map<String , String>  statsMap = new LinkedHashMap<>() ;

        double[] scores = studentScores.stream().mapToDouble(d -> d).toArray();
        double benchMark = 75.0 ;
        double mean = mean(scores) ;
        double variance = variance(scores);
        double std = Math.sqrt(variance) ;
        double HightestScore = max(scores);
        double LowestScore = min(scores);
        int numberOfStudents = studentScores.size();

        DescriptiveStatistics ds = new DescriptiveStatistics(scores);
        double median = ds.getPercentile(50);
        double firstQ = ds.getPercentile(25) ;
        double thirtQ = ds.getPercentile(75) ;

        double CI90Lower =Utils.getNumberWithinLimits( mean - calcCI(numberOfStudents,0.9 , std)  , 0 , maxScore) ;
        double CI90Higher = Utils.getNumberWithinLimits(mean + calcCI(numberOfStudents,0.9 , std)  , 0 , maxScore) ;

        double CI95Lower = Utils.getNumberWithinLimits(mean - calcCI(numberOfStudents,0.95 , std) , 0 , maxScore)  ;
        double CI95Higher = Utils.getNumberWithinLimits(mean + calcCI(numberOfStudents,0.95 , std)  , 0 , maxScore) ;

        double CI99Lower = Utils.getNumberWithinLimits(mean - calcCI(numberOfStudents,0.99 , std) , 0 , maxScore)  ;
        double CI99Higher = Utils.getNumberWithinLimits(mean + calcCI(numberOfStudents,0.99 , std)  , 0 , maxScore) ;

        double kr20  = calcKr20(formIndex);
        double kr21 = calcKr21(formIndex) ;
        double cronbachAlpha = calcCronbachAlpha(formIndex) ;


        statsMap.put("Number Of Students" , Utils.formatNumber(studentScores.size() , 0));

        statsMap.put("Number of Objective Questions" , Utils.formatNumber(numberOfQuestions , 0) );
        statsMap.put("Number of Subjective Questions" , Utils.formatNumber(subjMaxScores.size() , 0) );
        statsMap.put("Maximum Possible Score" , Utils.formatNumber(maxScore , 0)) ; // assuming all Forms should have the same weight sum
//        statsMap.put("Benchmark" ,  Utils.formatNumber(benchMark , 0 ) ) ;

        statsMap.put("Mean Score" , Utils.formatNumber(mean , 1 ));
        statsMap.put("Mean Percent Score" , Utils.formatNumber(mean/maxScore  * 100 ,1 )+"%" ) ;
        statsMap.put("Highest Score" ,Utils.formatNumber(HightestScore , 0) ) ;
        statsMap.put("Lowest Score" , Utils.formatNumber( LowestScore , 0 )) ;

        statsMap.put("Standard Deviation" , Utils.formatNumber( std , 1 ) ) ;
        statsMap.put("Variance" , Utils.formatNumber( variance,1)) ;
        statsMap.put("Range" , Utils.formatNumber( HightestScore - LowestScore , 1))  ;
        statsMap.put("Median" , Utils.formatNumber(median , 1));
        statsMap.put("25th Percentile" , Utils.formatNumber( firstQ , 1 )) ;
        statsMap.put("75th Percentile" , Utils.formatNumber( thirtQ  ,1 )) ;
        statsMap.put("Interquartile Range" , Utils.formatNumber( thirtQ-firstQ , 1 )) ;

        statsMap.put("90" , Utils.formatNumber( CI90Lower , 1 ) + " - " + Utils.formatNumber( CI90Higher , 1 )) ;
        statsMap.put("95" , Utils.formatNumber( CI95Lower , 1 ) + " - " + Utils.formatNumber(CI95Higher , 1 )) ;
        statsMap.put("99" , Utils.formatNumber( CI99Lower , 1 ) + " - " + Utils.formatNumber( CI99Higher , 1 )) ;

        statsMap.put("Kuder-Richardson Formula 20" ,  Utils.formatNumber( kr20 , 2 ))  ;
        statsMap.put("Kuder-Richardson Formula 21" ,  Utils.formatNumber( kr21 , 2)) ;
        statsMap.put("Cronbach's Alpha" , Utils.formatNumber(cronbachAlpha , 2));


        return statsMap ;

    }

    public static  ArrayList<Map<String ,String>> report3Stats() {

        int numberOfForms= getNumberOfForms() ;
        ArrayList<Map<String , String>> report3Maps = new ArrayList<>();

        Map <String , String > testInsightsMap ;
        Map <String , String > generalStatsMap ;
        Map compinedMap = new LinkedHashMap() ;

        double meanKr20  =0 ;
        double meanKr21 = 0 ;
        double meanCronbachAlpha = 0  ;
        if(numberOfForms>1) {
            for (int formIndex = 0; formIndex < numberOfForms; formIndex++) {
                generalStatsMap = calcGeneralStats(formsScors.get(formIndex), questionsChoices.size() , formIndex);
                testInsightsMap = calcFormTestInsights(correctAnswersPercents.get(formIndex));
                generalStatsMap.remove("Number Of Students");  // not in report 3
                compinedMap = new LinkedHashMap();
                compinedMap.putAll(testInsightsMap);
                compinedMap.putAll(generalStatsMap);
                report3Maps.add(compinedMap);

                meanKr20 += Double.valueOf(generalStatsMap.get("Kuder-Richardson Formula 20"));
                meanKr21 += Double.valueOf(generalStatsMap.get("Kuder-Richardson Formula 21"));
                meanCronbachAlpha += Double.valueOf(generalStatsMap.get("Cronbach's Alpha"));
            }

            meanKr20 /= numberOfForms ;
            meanKr21 /= numberOfForms;
            meanCronbachAlpha /= numberOfForms ;
        }

        testInsightsMap= calcTestInsights();
        generalStatsMap= calcGeneralStats(studentScores , questionsChoices.size() , 0 );
        generalStatsMap.remove("Number Of Students") ;  // not in report 3

        if(getNumberOfForms()>1) {
            generalStatsMap.put("Kuder-Richardson Formula 20", Utils.formatNumber(meanKr20, 2));
            generalStatsMap.put("Kuder-Richardson Formula 21", Utils.formatNumber(meanKr21, 2));
            generalStatsMap.put("Cronbach's Alpha", Utils.formatNumber(meanCronbachAlpha, 2));
        }
        compinedMap = new LinkedHashMap( );
        compinedMap.putAll(testInsightsMap);
        compinedMap.putAll(generalStatsMap);
        report3Maps.add(compinedMap);

        return report3Maps ;

    }
    private static void  initPointBiserialTables ()  {

        double[]forms = studentForms.stream().mapToDouble(d -> d).toArray();
        int numberOfForms = (int)max(forms)+1 ;

        birnaryStudentResponses = new ArrayList<ArrayList<ArrayList<Integer>>>();
        studentsCorrectAnswersCount= new ArrayList<ArrayList<Integer>>();

        for(int i = 0 ; i <numberOfForms ; i ++) {
            ArrayList<ArrayList<Integer>> table = new ArrayList<ArrayList<Integer>>() ;
            birnaryStudentResponses.add(table);

            ArrayList<Integer> formTotal = new ArrayList<Integer>() ;
            studentsCorrectAnswersCount.add(formTotal);
        }

        for (int studentIndex = 0 ; studentIndex < studentScores.size() ; studentIndex++) {
           int studentForm =  studentForms.get(studentIndex) ;
           int studentTotal =0 ;
           ArrayList<Integer> isStudentAnswersCorrect = new ArrayList<>(); // array for each question = 1 if student answered correctly else = 0
           ArrayList<String> studentAnswer =  studentAnswers.get(studentIndex);
           for ( int questionIndex =0 ; questionIndex<studentAnswer.size() ; questionIndex++) {
               String correctAnswer = correctAnswers.get(studentForm).get(questionIndex) ;
               int isStudentAnswerCorrect  = studentAnswer.get(questionIndex).equals(correctAnswer)?1:0 ;
               studentTotal+= isStudentAnswerCorrect ;
               isStudentAnswersCorrect.add(isStudentAnswerCorrect);
           }

            birnaryStudentResponses.get(studentForm).add(isStudentAnswersCorrect) ;
            studentsCorrectAnswersCount.get(studentForm).add(studentTotal) ;
        }

        for(int i = 0 ; i < numberOfForms ; i++) {
            birnaryStudentResponses.set(i ,  Utils.transpose(birnaryStudentResponses.get(i)));
        }
    }


    private static double calcPointBiserial(int formIndex , int questionIndex ) {
        double[] questionVector = birnaryStudentResponses.get(formIndex).get(questionIndex).stream().mapToDouble(d -> d).toArray();
        double[] totalVector = studentsCorrectAnswersCount.get(formIndex).stream().mapToDouble(d -> d).toArray();
        PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation() ;
        double pointBiserial =  pearsonsCorrelation.correlation(totalVector , questionVector) ;
        return Double.isNaN(pointBiserial)?0:pointBiserial;
    }

    private static String calcPrecentOfSolvers(double startPercent , double endPercent , int formIndex , int questionIndex ) {


        int totalNumberOfStudents = formsScors.get(formIndex).size();

        if(totalNumberOfStudents<3)
            return "-" ;

        ArrayList<ArrayList<String> > formSortedStudentAnswers = sortedStudentAnswers.get(formIndex);
        String correctAnswer = correctAnswers.get(formIndex).get(questionIndex);
        double count = 0  ;
        int studentStartIndex = (int)Math.round(startPercent*totalNumberOfStudents) ;
        int studentEndIndex = (int) Math.round(endPercent * totalNumberOfStudents) ;

        for (int studentIndex =  studentStartIndex ; studentIndex<  studentEndIndex  ; studentIndex++) {
           if(formSortedStudentAnswers.get(studentIndex).get(questionIndex).equals(correctAnswer))
               count++;
        }
        int numberOfStudents  = studentEndIndex - studentStartIndex ;

        return Utils.formatNumber( (double)count/(double)numberOfStudents *100 , 1 )+"%" ;
    }

    public static Map<String , String> report2GeneralStats(int formIndex) {
        return calcGeneralStats (formsScors.get(formIndex) , questionsChoices.size() , formIndex);
    }

    // this fucntion needs refactoring
    public static ArrayList<ArrayList<ArrayList<String>>> report2PrintableStats ( ArrayList<ArrayList<ArrayList<String>>> tablesStats , int formIndex) {

        ArrayList<ArrayList<ArrayList<String>>> newTables = new ArrayList<>() ;
        int questionIndex = 0 ;
        for(int tableIndex = 0  ; tableIndex < tablesStats.size() ; tableIndex++ ) {
            ArrayList<ArrayList<String>> tableStats = Utils.cloneTable(tablesStats.get(tableIndex));
            for (int rowIndex = 0 ; rowIndex < tableStats.size() ; rowIndex++) {
                ArrayList<String> tableRow = tableStats.get(rowIndex);
                String correctAnswer = correctAnswers.get(formIndex).get(questionIndex) ;
                int numberOfChoices = questionsChoices.get(questionIndex).size() ;
//
                String nonDistractors = "" ;
                for(int colIndex = 0 ; colIndex< tableRow.size();  colIndex++) {
                    String data = tableRow.get(colIndex);
                    if(colIndex>2 && colIndex < 3+numberOfChoices) {
                        if(data.contains(";")) {
                            String [] parts = data.split(";");
                            String cellData = parts[0] ;
                            String cellClass = parts[1] ;
                            //check for nonDistractor
                            if(cellData.equals("0")){
                                nonDistractors+= questionsChoices.get(questionIndex).get(colIndex-3)+" " ;
                            }
                            //remove colors
                            if (cellClass.equals("red")) {
                                cellData+="*;bold" ;
                            }
                            if (cellClass.equals("green")) {
                                cellData+=";under-line";
                            }
                            tableRow.set(colIndex , cellData);

                        }
                    }
                }
//                tableRow.add(2 , correctAnswer) ; //Correct Answers
                if(nonDistractors.equals(""))
                    nonDistractors="-";
//                tableRow.add(3+numberOfChoices , nonDistractors ) ; // nonDistractors
                questionIndex++ ;
            }
            newTables.add(tableStats) ;
        }
        return newTables ;
    }

    public static ArrayList<ArrayList<ArrayList<String>>> report2TableStats (int formIndex) {


        ArrayList<ArrayList<Double>> formStats = answersStats.get(formIndex) ;
        ArrayList<String> formCorrectAnswers= correctAnswers.get(formIndex);
        ArrayList<ArrayList<ArrayList<String>>> tables = new ArrayList<ArrayList<ArrayList<String>>>();
        ArrayList<ArrayList<String>> table = new ArrayList<ArrayList<String>>() ;


        ArrayList<Group> groups = CSVHandler.getDetectedGroups() ;
        int groupIndex = 0 ;
        int changeTableNumberOfQuestions = groups.get(0).getqCount() ;

        for(int questionIndex = 0 ; questionIndex < formCorrectAnswers.size() ; questionIndex++) {


            ArrayList<String> questionChoices = questionsChoices.get(questionIndex);
//            ArrayList<String> previousQuestionChoices = questionIndex>0?questionsChoices.get(questionIndex-1):questionChoices;

            // check if the table format changed ... create new table
            if(questionIndex == changeTableNumberOfQuestions) {
               tables.add(table);
               table = new ArrayList<ArrayList<String>>() ;
               changeTableNumberOfQuestions += groups.get(++groupIndex).getqCount() ;
            }
            ArrayList<String>tableRow = new ArrayList<>() ;
            tableRow.add(String.valueOf(questionIndex+1)) ; // NO.
            tableRow.add(questionNames.get(questionIndex));// Question
            ArrayList<Double> questionStats =  answersStats.get(formIndex).get(questionIndex) ;
            String correctAnswer = correctAnswers.get(formIndex).get(questionIndex) ;

            tableRow.add(correctAnswer); // correct answer
            int correctAnswerIndex = questionsChoices.get(questionIndex).indexOf(correctAnswer);

            double correctAnswerPrecentage = questionStats.get(correctAnswerIndex);

            String nonDistractors ="" ;
            for(int answerIndex = 0 ; answerIndex < questionStats.size() ; answerIndex ++ ) {
                String addedClass = "" ;
                String percentOfSolvers = Utils.formatNumber(questionStats.get(answerIndex) * 100 , 1) ;
                if(answerIndex== correctAnswerIndex)
                    addedClass=";green bold";
                else if(questionStats.get(answerIndex)> correctAnswerPrecentage) {
                    addedClass = ";red bold";
                    if(getMaxNumOfChoices() < Report2.MAX_ACCEPTABLE_CHOICES)
                        percentOfSolvers+= "*" ;
                }
                else if(questionStats.get(answerIndex) ==0 ) {
                    addedClass = ";gold";
                    if(nonDistractors.length() < Report2.MAX_NON_DESTRACOTRS *2 )
                        nonDistractors += questionChoices.get(answerIndex)+" " ;
//                    percentOfSolvers = "(0)";
                }
                tableRow.add(percentOfSolvers+addedClass) ; //Response Frequences
            }

            if(nonDistractors.equals(""))
                nonDistractors= "-" ;
            else if(nonDistractors.length()>= Report2.MAX_NON_DESTRACOTRS * 2)
                nonDistractors+= "..." ;
            tableRow.add(nonDistractors);
            tableRow.add(Utils.formatNumber(calcPointBiserial(formIndex, questionIndex) , 1)) ; // Point Biserial
            tableRow.add(Utils.formatNumber(correctAnswerPrecentage * 100 ,1 )+"%") ; // Total
            tableRow.add(calcPrecentOfSolvers(.75 , 1.0,formIndex , questionIndex)); //upper 27
            tableRow.add(calcPrecentOfSolvers(0 , .25,formIndex , questionIndex)); // lower 27
            table.add(tableRow);
        }
        tables.add(table) ;

//        System.out.println("el table ya man "+ tables);
        return tables ;
    }

    public static ArrayList<ArrayList<ArrayList<String>>> report5stats (int formIndex ){

        int numberOfStudents = formsScors.get(formIndex).size() ;


        ArrayList<ArrayList<Double>> formStats = answersStats.get(formIndex) ;
        ArrayList<String> formCorrectAnswers= correctAnswers.get(formIndex);
        ArrayList<ArrayList<ArrayList<String>>> tables = new ArrayList<ArrayList<ArrayList<String>>>();

        for(int questionIndex = 0 ; questionIndex < formCorrectAnswers.size() ; questionIndex++ ){
            ArrayList<ArrayList<String>> table = new ArrayList<ArrayList<String>>() ;
            ArrayList<String> questionChoices = questionsChoices.get(questionIndex) ;
            ArrayList<Double> questionStats = formStats.get(questionIndex) ;
            String correctAnswer = formCorrectAnswers.get(questionIndex);
            int correctAnswerIndex = questionChoices.indexOf(correctAnswer);
            double correctAnswerPrecentage = questionStats.get(correctAnswerIndex);
            for(int choiceIndex = 0 ; choiceIndex< questionChoices.size() ; choiceIndex ++) {
                ArrayList<String>tableRow = new ArrayList<>() ;
                String addedClass = "; " ;
                String barClass = "grayBar" ;
                String imgName = "" ;
                if(correctAnswerIndex == choiceIndex) {
                    addedClass = ";correct-answer";
                    barClass = "greenBar" ;
                    imgName = "correctColored";
                }
                tableRow.add(questionChoices.get(choiceIndex) + addedClass) ;
                tableRow.add(Utils.formatNumber(questionStats.get(choiceIndex)*numberOfStudents , 0 )) ;
                tableRow.add(Utils.formatNumber(questionStats.get(choiceIndex)*100 , 1 )) ;
                if(questionStats.get(choiceIndex)> correctAnswerPrecentage) {
                    barClass = "redBar";
                    imgName = "distractorColored";
                }
                if(questionStats.get(choiceIndex)==0)
                    imgName = "nonDistractorColored";

                String addedCol = imgName.equals("")?" ":"<img src='"+imgName+".png' height='15px' class='type-img'> </img>" ;
                tableRow.add(addedCol);
                tableRow.add(barClass);
                table.add(tableRow);
            }
            tables.add(table);
        }
        return tables ;
    }


    private static String getGrade(double scorePrecentage) {
        for (int gradeIndex = 0 ; gradeIndex < grades.size() ; gradeIndex ++) {
            if(gradesLowerRange.get(gradeIndex)> scorePrecentage)
                return grades.get(gradeIndex-1) ;
        }
        return grades.get(grades.size()-1);
    }


    public static ArrayList<ArrayList<String>> report4Stats( ){

        ArrayList<ArrayList<String>> statsTable = new ArrayList<ArrayList<String>>();

        double[] scores = studentScores.stream().mapToDouble(d -> d).toArray();



        for(int studentIndex  = 0 ; studentIndex < studentScores.size() ; studentIndex++ ) {
            ArrayList<String> tableRow = new ArrayList<String>() ;
            double scorePrecentage = studentScores.get(studentIndex)/ maxScore ;
            String studentId  = studentIdentifier.get(studentIndex) ;
            if(studentId.length()==0)
                tableRow.add("-") ;
            else
                tableRow.add(studentId) ; // identifier
            tableRow.add(getGrade(scorePrecentage)) ; // Grade
            tableRow.add(Utils.formatNumber(studentScores.get(studentIndex) , 0)+"/"+ Utils.formatNumber(maxScore , 0 )) ; // score
            tableRow.add(Utils.formatNumber(scorePrecentage*100 , 1)+"%") ; // percnetage


            statsTable.add(tableRow);
        }
        double meanScore = mean(scores) ;
        double meanScorePercentage = meanScore / maxScore ;
        ArrayList<String> meanRow = new ArrayList<String>() ;
        meanRow.add("mean;bold");
        meanRow.add(getGrade(meanScorePercentage)) ; // grade
        meanRow.add(Utils.formatNumber(meanScore, 0 )+"/"+Utils.formatNumber(maxScore , 0)) ; // score
        meanRow.add(Utils.formatNumber( meanScorePercentage*100 , 1 ) +"%") ;// score percentage

        statsTable.add(meanRow);
        return statsTable ;
    }



    public static ArrayList<ArrayList<String>> report1Stats( ) {



        ArrayList<ArrayList<String>> statsTable = new ArrayList<ArrayList<String>>();

        int numberOfStudents = studentScores.size();

        Map<String, Integer> gradesCount = new HashMap<>();

        //initialize all counts to zero
        for(int i = grades.size()-1 ; i>=0 ; i-- ) {
            gradesCount.put(grades.get(i) , 0) ;
        }

        for(Double score: studentScores) {
           String grade =  getGrade(score/maxScore) ;
           gradesCount.put(grade , gradesCount.get(grade) +1) ;
        }

        for(int gradeIndex = grades.size()-1 ; gradeIndex>=0 ; gradeIndex-- ) {
            ArrayList<String>tableRow  = new ArrayList<>() ;
            int gradeCount = gradesCount.get(grades.get(gradeIndex));
            tableRow.add(grades.get(gradeIndex)); // Grade
            tableRow.add(Utils.formatNumber(gradesLowerRange.get(gradeIndex)*100 , 0 )+
                    " - " + Utils.formatNumber(gradesLowerRange.get(gradeIndex+1)*100 , 0 ) ); // grade percentage range

            String gradeLowerRange = Utils.formatNumber(gradesLowerRange.get(gradeIndex)*maxScore , 1 ) ;
            gradeLowerRange = gradeLowerRange.contains(".")?gradeLowerRange:gradeLowerRange+".0" ;
            String gradeUpperRange =Utils.formatNumber(gradesLowerRange.get(gradeIndex+1)*maxScore , 1) ;
            gradeUpperRange = gradeUpperRange.contains(".")?gradeUpperRange:gradeUpperRange+".0" ;

            tableRow.add(gradeLowerRange+ " - " + gradeUpperRange) ; // grade score range
            tableRow.add(Utils.formatNumber(gradeCount , 0)) ; // number of students who got the grade
            tableRow.add(Utils.formatNumber( (double) gradeCount/(double)numberOfStudents *100 , 1) + "%"); // percentage of students who got the grade
            statsTable.add(tableRow);
        }
        return statsTable ;
    }

    /**
     * get the hardest question given a list of correct Answer Percent for each Question
     * @param correctPercents List of correct Answer Percent for each Question
     * @param startingIndex starting index of the list in the main list of questions
     * @return hardest question name
     */
    private static String getHardestQusetion (ArrayList<Double> correctPercents , int startingIndex){
        double[] percents = correctPercents.stream().mapToDouble(d -> d).toArray();
        double minPercent = min(percents) ;
        int questionIndex = correctPercents.indexOf(minPercent) + startingIndex ;
        return questionNames.get(questionIndex);
    }

    /**
     * get the easiest question given a list of correct Answer Percent for each Question
     * @param correctPercents List of correct Answer Percent for each Question
     * @param startingIndex starting index of the list in the main list of questions
     * @return easiest question name
     */
    private static String getEasiestQusetion (ArrayList<Double> correctPercents , int startingIndex){
        double[] percents = correctPercents.stream().mapToDouble(d -> d).toArray();
        double maxPercent = max(percents) ;
        int questionIndex = correctPercents.indexOf(maxPercent) + startingIndex ;
        return questionNames.get(questionIndex);
    }


    public static ArrayList<ArrayList<ArrayList<String>>> report6Stats () {

        ArrayList<ArrayList<ArrayList<String>>> report6FormTables = new ArrayList<>();
        ArrayList<ArrayList<String>> formTable = new ArrayList<>( );
        ArrayList<String> tableRow = new ArrayList<>( );

        for( int formIndex = 0 ; formIndex< getNumberOfForms() ; formIndex++ ) {
            ArrayList<Double> formCorrectPercents = correctAnswersPercents.get(formIndex);
            ArrayList<Group> groups = CSVHandler.getDetectedGroups();
            int qIndex = 0;
            for (Group group : groups) {
                tableRow.add(group.getCleanedName()); // group name
                int qCount = group.getqCount();

                ArrayList<Double> groupCorrectPercents = new ArrayList<>(formCorrectPercents.subList(qIndex, qIndex + qCount));
                String hardestQuestion = getHardestQusetion(groupCorrectPercents, qIndex);
                tableRow.add(hardestQuestion); // hardest question
                String easiestQuestion = getEasiestQusetion(groupCorrectPercents, qIndex);
                tableRow.add(easiestQuestion); // easiest question

                double[] percents = groupCorrectPercents.stream().mapToDouble(d -> d).toArray();
                double avgCorrectPercent = sum(percents) / percents.length;
                tableRow.add(Utils.formatNumber( avgCorrectPercent * 100 , 1) + "%"); // average correct percent

                Double pointBiserialSum = 0.0;
                int questionsWithDistractorCount = 0;
                for (int i = 0; i < qCount; i++) { // interate over each question in the group
                    pointBiserialSum += calcPointBiserial(formIndex, qIndex + i);
                    for (Double answerPercent : answersStats.get(formIndex).get(qIndex + i)) {  // iterate over percent of Responses for each question
                        if (answerPercent > formCorrectPercents.get(qIndex + i)) {  //check for distractors
                            questionsWithDistractorCount++;
                            break;
                        }
                    }
                }
                tableRow.add(Utils.formatNumber((double) questionsWithDistractorCount / qCount * 100 ,1 ) + "%"); // questions with Distractors
                tableRow.add(Utils.formatNumber(pointBiserialSum / qCount , 2)); // avg point biserial

                tableRow.add(Utils.formatNumber((1 - avgCorrectPercent) * 10 , 1)); // difficality
                formTable.add(tableRow);
                tableRow = new ArrayList<>();
                qIndex += qCount;
            }
            report6FormTables.add(formTable) ;
            formTable = new ArrayList<>();
        }

//        System.out.println(formTable);
        return  report6FormTables ;
    }

    private static  String getSmartDistractors (int formIndex , int questionIndex) {
        String smartDist = "" ;

        int totalNumberOfStudents = formsScors.get(formIndex).size();

        if(totalNumberOfStudents<3)
            return "-" ;

        ArrayList<ArrayList<String> > formSortedStudentAnswers = sortedStudentAnswers.get(formIndex);
        String correctAnswer = correctAnswers.get(formIndex).get(questionIndex);

        int studentStartIndex = (int)Math.round(.75*totalNumberOfStudents) ;
        int studentEndIndex = totalNumberOfStudents ;

        Set<String> smartDistSet = new HashSet<>() ;
        for (int studentIndex =  studentStartIndex ; studentIndex<  studentEndIndex  ; studentIndex++) {
            String stundentAnswer = formSortedStudentAnswers.get(studentIndex).get(questionIndex) ;
            if (!stundentAnswer.equals(correctAnswer))
                smartDistSet.add(stundentAnswer);

        }
        for (String dist: smartDistSet)
            smartDist+= dist +" " ;

        smartDist = Utils.removeLastChar(smartDist) ;
        return  smartDist ;
    }

    public static ArrayList<ArrayList<ArrayList<ArrayList<String>>>> report7Stats () {
        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> formsTableStats =  new ArrayList<>() ;

        for (int formIndex = 0 ; formIndex < getNumberOfForms() ; formIndex++) {
            ArrayList<ArrayList<ArrayList<String>>> oneFormTables = new ArrayList<>();

            ArrayList<ArrayList<String>> badQuestionsTable = new ArrayList<>();
            ArrayList<ArrayList<String>> questionsTable = new ArrayList<>();

            for (int questionIndex = 0; questionIndex < questionsChoices.size(); questionIndex++) {
                String correctAnswer = correctAnswers.get(formIndex).get(questionIndex);

                int correctAnswerIndex = questionsChoices.get(questionIndex).indexOf(correctAnswer);

                ArrayList<Double> questionStats = answersStats.get(formIndex).get(questionIndex);

                double correctAnswerPrecentage = questionStats.get(correctAnswerIndex);

                double pointBiserial = calcPointBiserial(formIndex, questionIndex);
                if (pointBiserial < 0) {

                    ArrayList<String> tableRow = new ArrayList<>();


                    tableRow.add(questionNames.get(questionIndex));
                    tableRow.add(Utils.formatNumber( pointBiserial  ,2 ));
                    tableRow.add(getSmartDistractors(formIndex, questionIndex));
                    tableRow.add(Utils.formatNumber( correctAnswerPrecentage * 100 , 1) + "%"); // Total
                    tableRow.add(calcPrecentOfSolvers(.75, 1.0, formIndex, questionIndex)); //upper 27
                    tableRow.add(calcPrecentOfSolvers(0, .25, formIndex, questionIndex)); // lower 27
                    badQuestionsTable.add(tableRow);
                }

                ArrayList<String> questionsRow = new ArrayList<>();
                questionsRow.add(questionNames.get(questionIndex));
                questionsRow.add(Utils.formatNumber( (1 - correctAnswerPrecentage) * 10 , 1 ));
                String distractors = "";
                String nonDistractors = "";
                for (int responseIndex = 0; responseIndex < questionStats.size(); responseIndex++) {
                    double responsePercent = questionStats.get(responseIndex);
                    if (responsePercent > correctAnswerPrecentage)
                        distractors += questionsChoices.get(questionIndex).get(responseIndex) + " ";
                    if (responsePercent == 0)
                        nonDistractors += questionsChoices.get(questionIndex).get(responseIndex) + " ";
                }


                questionsRow.add(Utils.removeLastChar(distractors));
                questionsRow.add(Utils.removeLastChar(nonDistractors));
                questionsRow.add(Utils.formatNumber( correctAnswerPrecentage * 100 , 1 )+"%");
                questionsTable.add(questionsRow);
            }


            SortByCorrectPercentsAsc sorterAsc = new SortByCorrectPercentsAsc();
            Collections.sort(questionsTable, sorterAsc);
            ArrayList<ArrayList<String>> hardestQuestionsTable ;

            ArrayList<ArrayList<String>> easiestQuestionsTable = new ArrayList<>() ;
            if(questionsTable.size()>10)
                hardestQuestionsTable = new ArrayList<>(Utils.removeTableCol(questionsTable, 3).subList(0, 10));
            else
                hardestQuestionsTable = Utils.removeTableCol(Utils.cloneTable(questionsTable) ,3) ;

            int counter = 0 ;
            for(int i = questionsTable.size()-1 ; i >-1 ; i--) {
                easiestQuestionsTable.add(questionsTable.get(i));
                if(counter == 10)
                    break;
                else
                    counter++ ;
            }
            easiestQuestionsTable = Utils.removeTableCol(Utils.cloneTable(easiestQuestionsTable) , 2) ;

            SortByPointBiserialAsc pointBiserialSorter = new SortByPointBiserialAsc() ;
            Collections.sort(badQuestionsTable , pointBiserialSorter);
            if(badQuestionsTable.size()>7) // if more than ten question return the worst 10 questions
                badQuestionsTable = new ArrayList<>(badQuestionsTable.subList(0,7)) ;

//            SortByCorrectPercentsDesc sorterDesc = new SortByCorrectPercentsDesc();
//            Collections.sort(questionsTable, sorterDesc);
//
//            if(questionsTable.size()>10)
//                hardestQuestionsTable = new ArrayList<>(Utils.removeTableCol(questionsTable, 3).subList(0, 10));
//            else
//                hardestQuestionsTable = Utils.removeTableCol(Utils.cloneTable(questionsTable) , 3) ;

            oneFormTables.add(hardestQuestionsTable);
            oneFormTables.add(easiestQuestionsTable);
            oneFormTables.add(badQuestionsTable);

            formsTableStats.add(oneFormTables) ;
        }
        return formsTableStats ;
    }

    public static ArrayList<ArrayList<ArrayList<ArrayList<String>>>> report9Stats() {

        double count;
        int studentsCount=studentAnswers.size();
        ArrayList<ArrayList<Double>> questAnswerStats = new ArrayList<>() ;
        for(int questionIndex=0;questionIndex<questionsChoices.size();questionIndex++){
            questAnswerStats.add(new ArrayList<Double>());
            for(String choice: questionsChoices.get(questionIndex)){
                count=0;
                for(int studentIndex=0;studentIndex<studentAnswers.size();studentIndex++){
                    if(studentAnswers.get(studentIndex).get(questionIndex).equals(choice) )
                        count++;
                }
                questAnswerStats.get(questionIndex).add(count/studentsCount);
            }
        }

        ArrayList<ArrayList<ArrayList<ArrayList<String>>>> tables = new ArrayList();

        ArrayList<Group> groups = CSVHandler.getDetectedGroups() ;
        int questionIndex = 0 ;
        for( int gIndex = 0 ; gIndex < groups.size() ; gIndex++ ) {

            ArrayList<ArrayList<ArrayList<String>>> groupTables = new ArrayList<>();

            for(int i = 0 ; i < groups.get(gIndex).getqCount() ; i++ ) {
                ArrayList<ArrayList<String>> table = new ArrayList<ArrayList<String>>();
                ArrayList<String> questionChoices = questionsChoices.get(questionIndex);
                ArrayList<Double> questionStats = questAnswerStats.get(questionIndex);
                for (int choiceIndex = 0; choiceIndex < questionChoices.size(); choiceIndex++) {
                    ArrayList<String> tableRow = new ArrayList<>();
                    tableRow.add(questionChoices.get(choiceIndex)); // response
                    tableRow.add(Utils.formatNumber(questionStats.get(choiceIndex) * studentsCount, 0)); // number of students
                    tableRow.add(Utils.formatNumber(questionStats.get(choiceIndex) * 100, 1));// percentage of students
                    tableRow.add("greenBar") ; // bar class
                    table.add(tableRow);
                }
                questionIndex++ ;
                groupTables.add(table) ;
            }
            tables.add(groupTables);
        }
        return tables ;
    }
    private static Pair<Double , Double> getTrendData (ArrayList<Double> hardness) {
        SimpleRegression regression = new SimpleRegression() ;

        double[] hardnessArray = hardness.stream().mapToDouble(d -> d).toArray();
        double maxHardness = max(hardnessArray);

        for(int i = 0 ; i < hardness.size() ; i ++) {
            regression.addData((double)i/(hardness.size()-1) , hardness.get(i)/maxHardness);
        }
        return new Pair<Double, Double>(regression.getSlope() , Math.sqrt(regression.getMeanSquareError())) ;
    }

    private static  double calcHarMean (double x , double y ) {
        return  2 * (x * y) / ( x + y) ;
    }
    private static double calcJowilParam (double slope , double error) {
        double slopeSign ;
        if(slope == 0)
            slopeSign = 1 ;
        else
            slopeSign = slope/Math.abs(slope) ;
        double harMean = calcHarMean(Math.abs(slope) , 1-Math.abs(error)) ;
        return (slopeSign* harMean + 1)*5;
    }

    private static ArrayList<Double> addTrendDataToList( ArrayList<Double> hardness ) {
        ArrayList<Double> graphData = (ArrayList)hardness.clone() ;
        Pair<Double,Double> trendData  = getTrendData(graphData) ;
        double slope = trendData.getKey() ;
        double error = trendData.getValue() *2 ; // multiply by to so that this value will be 0-1
        double jowil = calcJowilParam(slope , error) ;
        graphData.add(slope) ; graphData.add(error); graphData.add(jowil) ;
        return graphData  ;
    }

    public static ArrayList<ArrayList<ArrayList<Double>>> report8Stats() {
        ArrayList<ArrayList<ArrayList<Double>>> formsData = new ArrayList<>();

        for (int formIndex = 0 ; formIndex< getNumberOfForms() ; formIndex++ ) {
            ArrayList<ArrayList<Double>> formGraphsData = new ArrayList<>();

            ArrayList<Double> groupData = new ArrayList<>();
            ArrayList<Double> testData = new ArrayList<>();

            ArrayList<Double> formCorrectPercents = correctAnswersPercents.get(formIndex);


            ArrayList<Group> groups = CSVHandler.getDetectedGroups();
            int groupIndex = 0;
            int changeGroupQIndex = groups.get(0).getqCount();
            for (int questionIndex = 0; questionIndex < formCorrectPercents.size(); questionIndex++) {
                if (questionIndex == changeGroupQIndex) {
                    changeGroupQIndex += groups.get(++groupIndex).getqCount();
                    ArrayList<Double> groupGraphData = addTrendDataToList(groupData);
                    formGraphsData.add(groupGraphData);
                    groupData = new ArrayList<>();
                }
                double questionHardness = (1 - formCorrectPercents.get(questionIndex)) * 10;
                testData.add(questionHardness);
                groupData.add(questionHardness);
            }
            ArrayList<Double> groupGraphData = addTrendDataToList(groupData);
            formGraphsData.add(groupGraphData);

            ArrayList<Double> testGraphData = addTrendDataToList(testData);
            formGraphsData.add(0, testGraphData);

            formsData.add(formGraphsData);
        }
        return formsData ;
    }



}

class SortByScore implements Comparator<ArrayList<String>>
{
    public int compare(ArrayList<String> a, ArrayList<String> b)
    {
        double diff = Double.parseDouble(a.get(a.size()-1))-Double.parseDouble(b.get(b.size()-1)) ;
        if(diff<0)
            return -1 ;
        else if( diff == 0)
            return 0 ;
        else
            return 1 ;     }
}
class SortByCorrectPercentsAsc implements Comparator<ArrayList<String>>
{
    public int compare(ArrayList<String> a, ArrayList<String> b)
    {
        String s1 = a.get(4) ;
        String s2 = b.get(4) ;
        double diff = Double.parseDouble(s1.substring(0,s1.length()-1))-Double.parseDouble(s2.substring(0,s2.length()-1)) ;
        if(diff<0)
            return -1 ;
        else if( diff == 0)
            return 0 ;
        else
            return 1 ;     }
}
class SortByPointBiserialAsc implements Comparator<ArrayList<String>>
{
    public int compare(ArrayList<String> a, ArrayList<String> b)
    {
        String s1 = a.get(1) ;
        String s2 = b.get(1) ;
        double diff = Double.parseDouble(s1.substring(0,s1.length()-1))-Double.parseDouble(s2.substring(0,s2.length()-1)) ;

        if(diff<0)
            return -1 ;
        else if( diff == 0)
            return 0 ;
        else
            return 1 ;
    }
}