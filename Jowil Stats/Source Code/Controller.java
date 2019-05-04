package Jowil;

import com.jfoenix.controls.*;
import com.sun.javafx.scene.control.skin.TableHeaderRow;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Insets;
import javafx.geometry.NodeOrientation;
import javafx.geometry.Pos;
import javafx.geometry.Rectangle2D;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.stage.*;
import org.bouncycastle.util.Arrays;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.pdfsam.ui.RingProgressIndicator;

import java.io.*;
import java.net.URLDecoder;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.logging.Logger;
import static java.util.Arrays.asList;

public abstract class Controller {


    ///components

    @FXML
    protected Pane rootPane;
    protected Scene scene;
    protected Stage stage;
    protected Controller back;
    protected Controller next;
    protected JFXButton backButton= new JFXButton("Back");
    protected JFXButton nextButton=new JFXButton("Next");


    //data fields
    static public double resX;
    static public double resY;
    protected double rootWidth;
    protected double rootHeight;
    protected static final Logger logger = Logger.getLogger( FileConfigController.class.getName());
    final String myFXML;
    String myTitle;
    double XSCALE,YSCALE;
    final boolean isResizable;
    double navWidth;
    double navHeight;
    boolean isHeightCalling;
    final double DEFAULT_FONT_AWESOME_ICON_SIZE=resX*27/1280;
    protected double minWidth,minHeight;
    static boolean isTranslationMode=true;  //Arabic -> true
    static boolean isNormalScalingMode=true; // normal scaling true-> resX will be set to 1280 and resY to 680 so that all calculations are made in absolute pixels count and not relative to screen resolution (Note that the display used in development had visible resolution of 1280x680)
    static boolean isQuestMode; //Questionnaire
    static boolean isOpenMode;
    static boolean isDevMode; //the program isn't running in deployment environment.
    static boolean isTranslateFormContent; //choices and id is translated to arabic
    static boolean isAnswerKeyInFirstRow;


    BorderPane outerBorderPane=new BorderPane();
    AnchorPane topWrapperPane=new AnchorPane();
    ImageView progressImage=new ImageView();
    ImageView logoImage=new ImageView(new Image("Images/logojsSmall.png"));
    RingProgressIndicator stepCounterIndicator=new RingProgressIndicator(4);
    Label stepLabel;


    int stepIndex;
    String stepName="";




    static JSONObject gradeScalesJsonObj;
    HashMap<String,Double> classesFontSizes;
    public static HashMap<String,String> translations;

    double origSceneWidth,origSceneHeight;
    protected boolean isContentEdited=false;
    protected HBox buttonsHbox= new HBox();
    protected double headersFontSize;
    private boolean isMaximizedChanged=false;
    boolean isWait=false;

    boolean isBeginMaximised;
    boolean isStepWindow;
    static String projectName;

    static String SAVED_PROJECTS_FILE_NAME="Projects.json",TRANSLATIONS_FILE_NAME="Translations.json";
    static JSONObject savedProjectsJson,currentOpenedProjectJson,generalPrefsJson;;
    static String dataDirPath; //the data directory is generated in deployment build.xml file.

    static String selectedIdentifierName;
    static String selectedFormColName;
    final double CHECK_BOXES_SIZE=resX*14/1280;
    static final String MANUAL_MODE_INDICATOR=" (Manual Configuration)";
    static final String GRADE_SCALE_FILE_NAME = "GradeScales.json";
    static final String GENERAL_PREFS_FILE_NAME="GeneralPrefs.json",LAST_CSV_DIR_JSON_KEY="lastCSVDir";
    static final String Q_NAMES_JSON_KEY="qNames",OBJ_GROUPS_JSON_KEY="objGroups",Q_CHOICES_JSON_KEY="questionsChoices",
            OBJ_WEIGHTS_JSON_KEY="objWeights",SUBJ_WEIGHTS_JSON_KEY="subjWeights",SELECTED_SCALE_JSON_KEY="lastSelectedScale",
            REPORTS_CHOSEN_JSON_KEY="reportsChosen",FORMATS_CHOSEN_JSON_KEY="formatsChosen",REPORTS_OUT_PATH_JSON_KEY="reportsOutputDir",
            Q_COL_START_INDEX_JSON_KEY="qColStartIndex",Q_COL_END_INDEX_JSON_KEY="qColEndIndex",SUBJ_COL_START_INDEX_JSON_KEY="subjColStartIndex",
            SUBJ_COL_END_INDEX_JSON_KEY="subjColEndIndex", SUBJ_Q_COUNT_JSON_KEY="subjQuestionsCount",FORM_COL_INDEX_JSON_KEY="formColIndex",
            ID_COL_START_INDEX_JSON_KEY="idStartIndex",ID_COL_END_INDEX_JSON_KEY="idEndIndex",IS_MANUAL_MODE_JSON_KEY="isManualModeUsed",
            IS_RESPONSES_CONTAINS_HEADERS_JSON_KEY="isResponsesContainsHeaders",IS_ANSWER_KEY_CONTAINS_HEADERS_JSON_KEY="isAnswerKeyContainsHeaders",
            RESPONSES_FILE_PATH_JSON_KEY="responsesFilePath", ANSWERS_FILE_PATH_JSON_KEY="answersFilePath",IDENTIFIER_NAME_JSON_KEY="identifierName"
            ,FORM_COL_NAME_JSON_KEY="formColName",SAVED_RESPONSES_CSV_JSON_KEY="savedResponsesCSV",SAVED_ANSWER_KEY_CSV_JSON_KEY="savedAnswerKeyCSV",
            SAVED_INFO_HEADERS_JSON_KEY="infoHeaders",FORMS_COUNT_JSON_KEY="formsCount",PROJECT_NAME_JSON_KEY="name", ALLOW_EXCEED_FULL_MARK_JSON_KEY="allowExceedFullMark",
            BONUS_MARKS_JSON_KEY="bonusMarks",USER_MAX_SCORE_JSON_KEY="userMaxScore",SCALES_JSON_KEY="scales",IS_TRANSLATION_MODE_JSON_KEY="isTranslationMode",
            IS_NORMAL_SCALING_MODE_JSON_KEY="isNormalScalingMode",ADD_BONUS_TO_ALL_JSON_KEY="addBonusToAll",IS_QUEST_MODE_JSON_KEY="isQuestMode",
            ACTIVATION_KEY_JSON_KEY="activationKey",IS_FIRST_LAUNCH_JSON_KEY="isFirstLaunch",IS_TRANSLATE_FORM_CONTENT_JSON_KEY="isTranslateFormContent",
            ASK_CONTINUE_FILE_CONFIG_JSON_KEY="askContinueFileConfig",IS_ANSWER_KEY_IN_FIRST_ROW_JSON_KEY="isAnswerKeyInFirstRow";


    //Main methods

    Controller(String fxmlName, String myTitle, double XSCALE , double YSCALE, boolean isResizable,Controller back,String progressImageName,int stepIndex,String stepName,double minWidth,double minHeight){
        this(fxmlName,myTitle,XSCALE,YSCALE,isResizable,back,true,false,minWidth,minHeight);

        if(isTranslationMode)
            progressImageName=progressImageName.charAt(0)+"arabic"+".png";
        progressImage=new ImageView(new Image("Images/"+progressImageName));
        this.stepIndex=stepIndex;
        this.stepCounterIndicator.setProgress((int)(stepIndex/4.0*100));
        this.stepName=stepName;


    }



    Controller(String fxmlName, String myTitle, double XSCALE , double YSCALE, boolean isResizable,Controller back, boolean isStepWindow,boolean isMaximised,double minWidth,double minHeight){

        this.myFXML="/FXML/"+fxmlName;
        this.myTitle=myTitle;
        this.XSCALE=XSCALE;
        this.YSCALE=YSCALE;
        this.isResizable = isResizable;
        if(back==null)
            backButton.setVisible(false);
        this.back=back;
        this.isBeginMaximised=isMaximised;
        this.isStepWindow=isStepWindow;
        headersFontSize=resX*20/1280;
        this.minWidth=minWidth;
        this.minHeight=minHeight;
        initClassesFontSizes();
    }

    public static String getProjectName (){
        return projectName ;
    }
    public static boolean isIsTranslateFormContent() {
        return isTranslateFormContent;
    }

    protected abstract void initComponents();
    protected abstract void saveChanges();

    public void initialize() {

        stepCounterIndicator.setScaleX(0.2);
        stepCounterIndicator.setScaleY(0.2);
        int stepsCount=isQuestMode?2:4;
        this.stepLabel=new Label(constructMessage("Step ",Integer.toString(stepIndex+1)," of "+stepsCount+": ",stepName));
        //this.stepLabel=new Label(stepName);
        stepLabel.setStyle("-fx-font-weight: bold;-fx-font-size:"+resX*12/1280);


        if(isStepWindow)
            outerBorderPane.prefHeightProperty().bind(stage.heightProperty());
        else
            rootPane.prefHeightProperty().bind(stage.heightProperty());


        rootPane.prefWidthProperty().bind(stage.widthProperty());

        topWrapperPane.getChildren().addAll(progressImage,logoImage,stepLabel);
        topWrapperPane.setStyle("-fx-background-color:white;-fx-border-width: 0 0 1 0;-fx-border-color:#e8e8e8");
        outerBorderPane.setTop(topWrapperPane);

        initBackButton();
        initNextButton();
        initButtonsHBox();
        initComponents();
        buildComponentsGraph();
        stabalizeTables();

        if(isQuestMode)
            progressImage.setVisible(false);




    }

    public void showWindow(){
        stage.show();
        stage.setMaximized(false);
        rootPane.requestFocus();
        this.updateSizes();
    }

    public void startWindow(){
        try {

            // Hide this current window (if this is what you want)

            FXMLLoader loader = new FXMLLoader(getClass().getResource(myFXML));
            stage = new Stage();
            loader.setController(this);
            Pane root = loader.load();

            double sceneWidth=(resX==800 && this instanceof WeightsController)?673:resX/XSCALE; //673 to solve bar chart min width issue
            double sceneHeight=resY / YSCALE;

            origSceneWidth=sceneWidth;
            origSceneHeight=sceneHeight;

            if(isNormalScalingMode){
                Rectangle2D primaryScreenBounds = Screen.getPrimary().getVisualBounds();
                if(sceneWidth>=primaryScreenBounds.getWidth() && sceneHeight>=primaryScreenBounds.getHeight())
                    isBeginMaximised=true;
                else {
                    sceneWidth = Math.min(sceneWidth, primaryScreenBounds.getWidth());
                    sceneHeight = Math.min(sceneHeight, primaryScreenBounds.getHeight());
                }
            }

            if(isStepWindow){
                outerBorderPane.setCenter(rootPane);
                scene = new Scene(outerBorderPane, sceneWidth, sceneHeight);
            }
            else
                scene=new Scene(root,sceneWidth, sceneHeight);


            if(isStepWindow)
                stage.setTitle(isOpenMode?projectName:"New Project");
            else
                stage.setTitle(myTitle);

            stage.setMinWidth(Math.min(minWidth,sceneWidth));
            stage.setMinHeight(Math.min(minHeight,sceneHeight*1.05));


            scene.getStylesheets().add(getClass().getResource("/FXML/application.css").toExternalForm());
            stage.setScene(scene);
            stage.setResizable(isResizable);

            stage.maximizedProperty().addListener(new ChangeListener<Boolean>() {
                public void changed(ObservableValue<? extends Boolean> observableValue, Boolean old, Boolean neww) {
                    isMaximizedChanged=true;


                }
            });


            stage.widthProperty().addListener(new ChangeListener<Number>() {
                public void changed(ObservableValue<? extends Number> observableValue, Number oldSceneWidth, Number newSceneWidth) {

                    isHeightCalling=false;
                    updateSizes();

                }
            });



            stage.heightProperty().addListener(new ChangeListener<Number>() {
                public void changed(ObservableValue<? extends Number> observableValue, Number oldSceneHeight, Number newSceneHeight) {

                    isHeightCalling=true;
                    updateSizes();
                }
            });



            rootPane.requestFocus();
            rootPane.setOnMouseClicked(t->rootPane.requestFocus());

            if(isTranslationMode) {
                if(!(this instanceof StartController))
                    scene.getRoot().setNodeOrientation(NodeOrientation.RIGHT_TO_LEFT);
                if(translations.containsKey(stage.getTitle()))
                    stage.setTitle(translations.get(stage.getTitle()));

                //progressImage.setNodeOrientation(NodeOrientation.RIGHT_TO_LEFT);
            }

            if(this instanceof StartController)
                stage.initStyle(StageStyle.UNDECORATED);
            else
                stage.initModality(Modality.APPLICATION_MODAL);
            stage.setOnCloseRequest(event -> {
                String small=isOpenMode?"close":"cancel";
                String capital=isOpenMode?"Close":"Cancel";
                String extra=isOpenMode?"Changes you made will not be saved. ":"";
                if(!showConfirmationDialog(capital+" Project",constructMessage(extra,"Are you sure you want to ",small," this project?"),stage.getOwner()))
                    event.consume();
            });


            stage.getIcons().add(new Image("Images/jsIcon.png"));

            stage.show();
            updateControlsText();
            stage.setMaximized(isBeginMaximised);
            updateSizes();


        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
    protected void updateSizes(){

        Insets navPadding=new Insets(0,0,0,0);
        navWidth=resX/15;
        navHeight=resX/47.5;





        if(!isHeightCalling) {
            if (scene != null && !isMaximizedChanged) {
                rootWidth = scene.getWidth();
            } else { //if maximised property was changed use scene width will not be properly set
                rootWidth = rootPane.getPrefWidth();
                System.out.println("maxim");
                isMaximizedChanged = false;
            }
        }


        progressImage.setFitWidth(resX*0.25);
        progressImage.setFitHeight(progressImage.getFitWidth()*0.12);



        topWrapperPane.setPrefHeight(progressImage.getFitHeight());


        logoImage.setFitHeight(progressImage.getFitHeight()*0.6);
        logoImage.setFitWidth(logoImage.getFitHeight());

        double topSpace=topWrapperPane.getPrefHeight()*0.2;
        double sideSpace=resX*3/1280;
        AnchorPane.setLeftAnchor(progressImage,(rootWidth-progressImage.getFitWidth())/2);
        AnchorPane.setRightAnchor(logoImage,logoImage.getFitWidth()+sideSpace);
        AnchorPane.setTopAnchor(logoImage,topSpace);

        AnchorPane.setLeftAnchor(stepLabel,6.0);
        AnchorPane.setTopAnchor(stepLabel,topSpace);





        if(isStepWindow) {
            rootHeight = outerBorderPane.getPrefHeight() - topWrapperPane.getPrefHeight();
            rootPane.setPrefHeight(rootHeight);
        }
        else
            rootHeight=rootPane.getPrefHeight();


        backButton.setPrefWidth(navWidth);
        backButton.setPrefHeight(navHeight);
        backButton.setPadding(navPadding);

        nextButton.setPrefWidth(navWidth);
        nextButton.setPrefHeight(navHeight);
        nextButton.setPadding(navPadding);

        double buttonsHBoxShift=0.03;

        buttonsHbox.setLayoutX(rootWidth*buttonsHBoxShift);

        buttonsHbox.setPrefHeight(resY*680/35);

        double hpos=0.877;
        if(rootHeight<=resY/1.34)
            buttonsHbox.setLayoutY(rootHeight*hpos);
        else
            buttonsHbox.setLayoutY(rootHeight-resY*(1-hpos));



        buttonsHbox.setSpacing(resX/100);
        buttonsHbox.setPrefWidth(rootWidth*(1-buttonsHBoxShift*2));
        buttonsHbox.setPadding(new Insets(resY/100, 0, 0, 0));


        buttonsHbox.setAlignment(Pos.TOP_RIGHT);
        System.out.println(rootWidth);
        System.out.println(rootHeight);

//        if(isNormalScalingMode && scene!=null && rootWidth<minWidth)
//            updateControlsText(rootWidth/origSceneWidth);
//
//

    }



    //helper methods
    protected static void showAlert(Alert.AlertType alertType, javafx.stage.Window owner, String title, String message) {
        constructAlert(alertType,owner,title,message).show();
    }


    protected static void showAlertAndWait(Alert.AlertType alertType, javafx.stage.Window owner, String title, String message) {
        constructAlert(alertType,owner,title,message).showAndWait();
    }

    private static Alert constructAlert(Alert.AlertType alertType, javafx.stage.Window owner, String title, String message){
        Alert alert = new Alert(alertType);
        alert.getButtonTypes().setAll(ButtonType.CLOSE);
        Button closeButt=(Button)alert.getDialogPane().lookupButton(ButtonType.CLOSE);
        closeButt.setText("OK");
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.initOwner(owner);
        processDialog(alert);
        return alert;
    }


    /**
     *
     * @param s
     * @return null if the value is less than zero or is not Parsable to Double , otherwise returns a new string in a non-integer representation
     */
    public static String tryDouble(String s){

        s=s.trim();
        try
        {
            double x=Double.parseDouble(s);
            if(x>=0)
                return Double.toString(x);
            else
                return null;
        }
        catch(NumberFormatException e)
        {
            return null;
        }

    }

    private void initClassesFontSizes() {
        classesFontSizes=new HashMap<>();
        classesFontSizes.put(JFXTextField.class.getSimpleName(),resX*14/1280);
        classesFontSizes.put(TextField.class.getSimpleName(),resX*12/1280);
        classesFontSizes.put(JFXComboBox.class.getSimpleName(),resX*14/1280);
        classesFontSizes.put(JFXToggleButton.class.getSimpleName(),resX*12/1280);
        classesFontSizes.put(JFXButton.class.getSimpleName(),resX*12/1280);
        classesFontSizes.put(Button.class.getSimpleName(),resX*12/1280);
        classesFontSizes.put(JFXTreeTableView.class.getSimpleName(),resX*14/1280);
        classesFontSizes.put(TableView.class.getSimpleName(),resX*12/1280);
        classesFontSizes.put(JFXCheckBox.class.getSimpleName(),resX*12/1280);
    }


    private void initBackButton(){

        rootPane.getChildren().add(backButton);
        backButton.setOnMouseClicked(t->{
            rootPane.requestFocus();
            back.showWindow();
            stage.close();
        });


    }

    private void initButtonsHBox(){
        buttonsHbox.setStyle("-fx-border-width: 1 0 0 0;-fx-border-color:#A9A9A9");
        //buttonsHbox.setStyle("-fx-border-width: 1 0 0 0;-fx-border-color:#095c90");
    }

    protected void updateControlsText() {
        updateControlsText(1.0);

    }
    private void updateControlsText(double fontScalingFactor) {
        LinkedBlockingQueue<Parent> queue=new LinkedBlockingQueue<>();
        queue.add(scene.getRoot());
        while(!queue.isEmpty()){
            Parent parent=queue.poll();
            String pClassName=parent.getClass().getSimpleName();
            if(classesFontSizes.containsKey(pClassName))
                parent.setStyle("-fx-font-size:"+classesFontSizes.get(pClassName)*fontScalingFactor);




            if(isTranslationMode)
                tryTranslate(parent);

            for(Node node:parent.getChildrenUnmodifiable()){
                if(node instanceof Parent)
                    queue.add((Parent)node);
                else{ //leaf nodes
                    String nClassName=node.getClass().getSimpleName();
                    if(classesFontSizes.containsKey(nClassName))
                        node.setStyle("-fx-font-size:" + classesFontSizes.get(nClassName)*fontScalingFactor);

                    if(isTranslationMode)
                        tryTranslate(node);
                }
            }
        }
    }


    public static void tryTranslate(Node node){


        try {
            String newText;
            if (node instanceof Labeled && (newText = translations.get(((Labeled) node).getText())) != null)
                ((Labeled) node).setText(newText);
            else if (node instanceof TextInputControl && (newText = translations.get(((TextInputControl) node).getPromptText())) != null)
                ((TextInputControl) node).setPromptText(newText);
            else if (node instanceof ComboBox && (newText = translations.get(((ComboBox) node).getPromptText())) != null)
                ((ComboBox) node).setPromptText(newText);
        }catch(RuntimeException e){//setting bound text in dialogs can cause runtime exception
            e.printStackTrace();
        }

    }


    protected void initNextButton(){
        //nextButton.setStyle("-fx-border-width:1;-fx-border-color:#949797");

//        nextButton.setOnMouseEntered(t->nextButton.setStyle("-fx-background-color:#878a8a;"));
//        nextButton.setOnMouseExited(t->nextButton.setStyle("-fx-background-color:transparent;"));
        nextButton.setOnMouseClicked(t->{
            rootPane.requestFocus();
            goToNextWindow();});

    }





    protected abstract Controller getNextController();

    protected void goToNextWindow(){
        saveChanges();

        if(next==null || isContentEdited) { //if first time or edit manually has been pressed
            next = getNextController();
            next.startWindow();
        }
        else
            next.showWindow();

        isContentEdited=false;
        stage.close();
    }


    protected void buildComponentsGraph(){

        buttonsHbox.getChildren().add(backButton);
        buttonsHbox.getChildren().add(nextButton);
        rootPane.getChildren().add(buttonsHbox);
    }

    protected double resXToPixels(double relativeVal){

        return relativeVal*resX;
    }

    protected double resYToPixels(double relativeVal){

        return relativeVal*resY;
    }

    protected double rootWidthToPixels(double relativeVal){

        return relativeVal*rootWidth;
    }

    protected double rootHeightToPixels(double relativeVal){

        return relativeVal*rootHeight;
    }



    public static String getDataDirPath() {
        return dataDirPath;
    }

    protected void stabalizeTables(){

    }

    protected static void disableTableDrag(TableView table){
        try {
            table.setOnMouseEntered(t -> {
                TableHeaderRow header = (TableHeaderRow) table.lookup("TableHeaderRow");
                header.setMouseTransparent(true);
            });
        }catch(Exception e){

        }
    }

    public void setContentEdited(boolean contentEdited) {
        isContentEdited = contentEdited;

    }

    public static boolean showConfirmationDialog(String title,String content,Window owner) {

        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        Button okButt=(Button)alert.getDialogPane().lookupButton(ButtonType.OK);
        okButt.setText("Yes");

        Button cancelButt=(Button)alert.getDialogPane().lookupButton(ButtonType.CANCEL);
        cancelButt.setText("No");

        alert.initOwner(owner);

        processDialog(alert);
        Optional<ButtonType> option = alert.showAndWait();

        return option.get() == ButtonType.OK;
    }


    protected JSONObject loadJsonObj(String path) {

        String file = dataDirPath+"UserData/"+path;
        JSONObject jsonObj = null;

        try {
            jsonObj = (JSONObject) new JSONParser().parse(new InputStreamReader(new FileInputStream(file), "UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();

        } catch (ParseException e) {
            e.printStackTrace();
        }

        return jsonObj;

    }

    protected void saveJsonObj(String path, JSONObject jsonObj) {
        if(jsonObj==null)
            return;

        PrintWriter pw = null;
        String file = dataDirPath+"UserData/"+path;

        try {
            pw = new PrintWriter(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        pw.write(jsonObj.toJSONString());
        pw.flush();
        pw.close();
    }

    public static String constructMessage(String... subMessages){

        StringBuilder sb=new StringBuilder();

        for(String message:subMessages)
            sb.append(isTranslationMode&&translations.containsKey(message)?translations.get(message):message);

        return sb.toString();
    }


    public static void processDialog(Dialog dialog){


        ((Stage)(dialog.getDialogPane().getScene().getWindow())).getIcons().add(new Image("Images/jsIcon.png"));
        dialog.getDialogPane().getStylesheets().add(Controller.class.getResource("/FXML/application.css").toExternalForm());

        if(isTranslationMode){
            DialogPane dialogPane=dialog.getDialogPane();
            String title=dialog.getTitle();
            if(translations.containsKey(title))
                dialog.setTitle(translations.get(title));

            for(ButtonType buttonType:dialogPane.getButtonTypes()){
                Button butt=(Button)dialogPane.lookupButton(buttonType);
                tryTranslate(butt);
            }

            dialog.getDialogPane().setNodeOrientation(NodeOrientation.RIGHT_TO_LEFT);
            translateAllNodes(dialogPane);
        }

    }


    public static void translateAllNodes(Parent root) {
        if(!isTranslationMode)
            return;
        LinkedBlockingQueue<Parent> queue=new LinkedBlockingQueue<>();
        queue.add(root);
        while(!queue.isEmpty()){
            Parent parent=queue.poll();
            tryTranslate(parent);

            for(Node node:parent.getChildrenUnmodifiable()){
                if(node instanceof Parent)
                    queue.add((Parent)node);
                else
                    tryTranslate(node);
            }
        }
    }

    protected static boolean makeDir(String dirPath){

        File theDir = new File(dirPath);
        System.out.println("Dir "+ theDir.getAbsolutePath());

        if (!theDir.exists()) {

            try{
                return theDir.mkdirs();
            }
            catch(SecurityException se){
                return false;
            }

        }
        else
            return true;

    }




}
