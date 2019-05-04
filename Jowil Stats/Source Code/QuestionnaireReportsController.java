package Jowil;

import javafx.geometry.Insets;
import javafx.scene.control.Alert;

import java.util.ArrayList;

public class QuestionnaireReportsController extends GradeBoundariesController{

    QuestionnaireReportsController(Controller back) {
        super(back);
        this.myTitle="Reports";
        this.minWidth=0;
        this.minHeight=340;
        this.XSCALE=3;
        this.YSCALE=2.1;
        this.isStepWindow=false;
    }

    @Override
    protected void initComponents() {
        if( (gradeScalesJsonObj = loadJsonObj(GRADE_SCALE_FILE_NAME)) == null)
            showAlertAndWait(Alert.AlertType.ERROR, stage.getOwner(), "Grade Configurations Error",
                    "Error in loading Grade Scale Configurations.");
        super.initComponents();
        rootPane.getChildren().remove(comboHBox);
        rootPane.getChildren().remove(scrollPane);
        rootPane.getChildren().remove(gradesVBox);

        rootPane.getChildren().remove(gradeBoundariesTitle);
        rootPane.getChildren().remove(midSeparator);
        rootPane.getChildren().remove(reportsConfigTitle);

        rootPane.getChildren().remove(reportsConfigAnc);
        rootPane.getChildren().add(questReportsScrollPane);
        questReportsScrollPane.setContent(formatsVBox);
        formatsVBox.getChildren().remove(formatsMasterCheckBox);

        buttonsHbox.setStyle("-fx-border-width: 0 0 0 0;-fx-border-color:#A9A9A9");
        backButton.setVisible(false);
        nextButton.setText("OK");
        questReportsScrollPane.setStyle("-fx-background:white");

        addFakeWeights(CSVHandler.getDetectedQHeaders().size(),1);
    }

    @Override
    protected void goToNextWindow() {
        super.goToNextWindow();
        back.stage.close();
    }

        @Override
    protected void updateSizes() {
        super.updateSizes();
        reportsDirHBox.setLayoutX(buttonsHbox.getLayoutX());
        reportsDirHBox.setLayoutY(resY*0.05);
        reportsDirHBox.setPrefWidth(buttonsHbox.getPrefWidth());
        questReportsScrollPane.setLayoutX((int)reportsDirHBox.getLayoutX());
        questReportsScrollPane.setLayoutY((int)(resY*0.13));
        questReportsScrollPane.setPrefWidth((int)reportsDirHBox.getPrefWidth());
        questReportsScrollPane.setPrefHeight((int)rootHeight*0.5);
        formatsVBox.setPadding(new Insets((int)(questReportsScrollPane.getPrefHeight() * 0.05),0,0,
                (int)(reportsConfigAnc.getPrefWidth() * 0.02)));

        double hpos=0.877;
        buttonsHbox.setLayoutY(rootHeight-resY*(1-hpos));

    }

    @Override
    public void startWindow(){
        super.startWindow();
        stage.setOnCloseRequest(event->{
        });

    }

    public static void addFakeWeights(int numberOfQuestions , int numberOfForms){
        ArrayList<ArrayList<Double>> formsWeights = new ArrayList<>() ;
        ArrayList<Double> formWeights = new ArrayList<Double>();
        for (int i = 0; i < numberOfQuestions; i++)
            formWeights.add(1.0);

        for(int formIndex = 0 ; formIndex < numberOfForms ; formIndex++) {
            formsWeights.add(formWeights);
        }

        Statistics.setQuestionWeights(formsWeights);
    }
}
