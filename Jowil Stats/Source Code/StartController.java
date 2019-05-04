package Jowil;

import Jowil.Reports.Report;
import com.jfoenix.controls.*;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.*;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.control.Button;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.*;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.Screen;
import javafx.util.Callback;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.pdfsam.ui.RingProgressIndicator;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.io.*;
import java.net.URLDecoder;
import java.util.*;
import java.util.regex.Pattern;


public class StartController extends Controller{


    @FXML
    AnchorPane lowerAncPane;

    @FXML
    ImageView logoImageView;

    @FXML
    Label jowilLabel;

    @FXML
    StackPane openStack;

    @FXML
    Rectangle openRect;

    @FXML
    Label openLabel;

    @FXML
    ImageView openImageView;


    @FXML
    StackPane newStack;

    @FXML
    Rectangle newRect;

    @FXML
    Label newLabel;

    @FXML
    ImageView newImageView;

    final static double scalingFactor=1.03;

    ArrayList<String> projectsNames;


    @FXML
    StackPane closeButton;

    @FXML
    ImageView closeImageView;

    @FXML
    StackPane minusButton;

    @FXML
    StackPane settingsButton;

    @FXML
    ImageView backImageView;


    @FXML
    ImageView settingsImageView;

    ContextMenu settingsMenu=new ContextMenu();

    AnchorPane langAnc=new AnchorPane();
    Label langLabel=new Label("Language");
    RadioButton englishRadio=new RadioButton("English");
    RadioButton arabicRadio=new RadioButton("العربية");
    ToggleGroup langToggleGroup=new ToggleGroup();
    CustomMenuItem langMenuItem=new CustomMenuItem(langAnc);
    final double langAncWidth=180;

    AnchorPane resScalingAnc=new AnchorPane();
    Label resScalingLabel=new Label("Relative resolution scaling");
    JFXToggleButton resScalingToggle=new JFXToggleButton();
    CustomMenuItem resScalingMenuItem=new CustomMenuItem(resScalingAnc);
    String onString="On", offString="Off";
    double realResX=Screen.getPrimary().getVisualBounds().getWidth();
    double realResY=Screen.getPrimary().getVisualBounds().getHeight();






    ObservableList<String> existingProjectsListItems;


    StartController() {
        super("Start.fxml", "Jowil Stats", 1.25,1.25 , true, null,false,true,0,0);
        initDataDirPath();
        generalPrefsJson=loadJsonObj(GENERAL_PREFS_FILE_NAME);
        if(generalPrefsJson==null){ //failed to load general prefs
            isTranslationMode=false;
            isNormalScalingMode=true;
        }
        else{
            isTranslationMode=(Boolean)generalPrefsJson.get(IS_TRANSLATION_MODE_JSON_KEY);
            isNormalScalingMode=(Boolean)generalPrefsJson.get(IS_NORMAL_SCALING_MODE_JSON_KEY);
        }


        /*in normal scaling set resX and resY to the default resolution used on the development PC so that all controls and
        fonts will use absolute numbers of pixels otherwise resX and resY will use the realvalues of the
        screen x and y resolutions and all controls sizes and fonts will be a function of screen resolution.*/
        if(isNormalScalingMode){
            resX=1280;
            resY=680;
        }
        else{
            Rectangle2D primaryScreenBounds = Screen.getPrimary().getVisualBounds();
            resX=primaryScreenBounds.getWidth();
            resY=primaryScreenBounds.getHeight();
        }

        System.out.println("resX:"+resX+",\tresY:"+resY);

        loadTranslationsJson();

    }



    @Override
    protected void initComponents() {

        double buttFontSize=isNormalScalingMode && Screen.getPrimary().getVisualBounds().getWidth()<1280?11:resX*14/1280;  //prevent labels from being larger than rectangles
        Font buttonsFont=new Font("System Bold",buttFontSize);
        openLabel.setFont(buttonsFont);
        newLabel.setFont(buttonsFont);
        //lowerAncPane.setStyle("-fx-background-color:transparent;-fx-border-width:1 0 0 0;-fx-border-color:#626365"); //anchor pane not white

        newStack.setOnMouseEntered(event -> {

            newRect.setScaleX(scalingFactor);
            newRect.setScaleY(scalingFactor);
            newImageView.setScaleX(scalingFactor);
            newImageView.setScaleY(scalingFactor);
            Font scaledFont=new Font("System Bold",buttonsFont.getSize()*scalingFactor);
            newLabel.setFont(scaledFont);
        });
        newStack.setOnMouseExited(event -> {
            newRect.setScaleX(1);
            newRect.setScaleY(1);
            newImageView.setScaleX(1);
            newImageView.setScaleY(1);
            newLabel.setFont(buttonsFont);
        });

//        newStack.setOnMousePressed(event -> {
//            newRect.setScaleX(1);
//            newRect.setScaleY(1);
//            newImageView.setScaleX(1);
//            newImageView.setScaleY(1);
//            newLabel.setFont(buttonsFont);
//        });


        openStack.setOnMouseEntered(event -> {

            openRect.setScaleX(scalingFactor);
            openRect.setScaleY(scalingFactor);
            openImageView.setScaleX(scalingFactor);
            openImageView.setScaleY(scalingFactor);
            Font scaledFont=new Font("System Bold",buttonsFont.getSize()*scalingFactor);
            openLabel.setFont(scaledFont);
        });
        openStack.setOnMouseExited(event -> {
            openRect.setScaleX(1);
            openRect.setScaleY(1);
            openImageView.setScaleX(1);
            openImageView.setScaleY(1);
            openLabel.setFont(buttonsFont);
        });



        newStack.setOnMouseClicked(event -> showNewProjectNameDialog(""));
        openStack.setOnMouseClicked(event -> showExistingProjects());

        closeButton.setOnMouseClicked(event -> {
            saveJsonObj(GENERAL_PREFS_FILE_NAME,generalPrefsJson);
            Platform.exit();
        });
        closeButton.setOnMouseEntered(event -> {
            closeButton.setStyle("-fx-background-color:rgb(240,108,116)");
            closeImageView.setImage(new Image("Images/whiteClose.png"));

        });
        closeButton.setOnMouseExited(event -> {
            closeButton.setStyle("-fx-background-color:transparent");
            closeImageView.setImage(new Image("Images/blackClose.png"));

        });

        minusButton.setOnMouseClicked(event -> stage.setIconified(true));




        settingsButton.setOnMouseEntered(event -> settingsImageView.setImage(new Image("Images/blackSettings.png")));
        settingsButton.setOnMouseExited(event -> settingsImageView.setImage(new Image("Images/darkGreySettings.png")));
        settingsButton.setOnMouseClicked(event ->{
            settingsMenu.hide(); //if already open
            settingsMenu.show(settingsButton,settingsButton.getLayoutX()-langAncWidth+settingsImageView.getFitWidth()/2,settingsButton.getLayoutY()+35);
        });

        TranslatableTooltip tooltip = new TranslatableTooltip("Settings");
        Tooltip.install(settingsButton, tooltip);



        initSettingsMenu();



    }


    @Override
    public void startWindow(){
        super.startWindow();



        if((Boolean)generalPrefsJson.get(IS_FIRST_LAUNCH_JSON_KEY)){
            if(!showLanguageFirstTimeDialog(true)) {
                stage.close();
                return;
            }
        }


        if(!checkActivationKey()){
            stage.close();
            return;
        }



        loadProjectsJson();
        onString=isTranslationMode&& translations.containsKey(onString)?translations.get(onString):onString;
        offString=isTranslationMode&& translations.containsKey(offString)?translations.get(offString):offString;
        stage.setOnCloseRequest(event -> { //override parent behaviour

        });



    }

    private boolean showLanguageFirstTimeDialog(boolean isEnglishSelected) {
        Dialog<String> dialog = new Dialog<>();
        dialog.setTitle("Select Language");
        dialog.setHeaderText("Welcome to Jowil Stats!");
        dialog.setGraphic(null);

        double flagsSize=15;
        VBox radiosHBox=new VBox(10);
        radiosHBox.setPadding(new Insets(25,0,0,10));
        ToggleGroup toggGroup= new ToggleGroup();
        RadioButton engRadio=new JFXRadioButton("English");
        RadioButton araRadio=new JFXRadioButton("العربية");
        engRadio.setToggleGroup(toggGroup);
        ImageView englishImageView=new ImageView("Images/USA.png");
        englishImageView.setFitWidth(flagsSize);
        englishImageView.setFitHeight(flagsSize);
        //engRadio.setGraphic(englishImageView);
        engRadio.setPadding(new Insets(0,0,0,7));
        araRadio.setPadding(engRadio.getPadding());
        araRadio.setToggleGroup(toggGroup);
        toggGroup.selectToggle(isEnglishSelected?engRadio:araRadio);
        ImageView arabicImageView=new ImageView("Images/Egypt.png");
        arabicImageView.setFitWidth(flagsSize);
        arabicImageView.setFitHeight(flagsSize);
        //araRadio.setGraphic(arabicImageView);
        Label label=new Label("Language");
        label.setFont(new Font("System Bold",13.5));
        ImageView langImg=new ImageView("Images/Language_50px.png");
        langImg.setFitWidth(20);
        langImg.setFitHeight(20);
        label.setGraphic(langImg);
        radiosHBox.getChildren().addAll(label,engRadio,araRadio);

        radiosHBox.setMinWidth(300);
        radiosHBox.setMinHeight(130);


        dialog.getDialogPane().setContent(radiosHBox);
        ButtonType continueButt=new ButtonType("Continue");
        dialog.getDialogPane().getButtonTypes().setAll(continueButt);
        dialog.getDialogPane().lookupButton(continueButt).setStyle("-fx-border-color: #095c90;-fx-text-fill:#095c90;");


        dialog.setY(realResY*0.26);
        dialog.setX(realResX*0.375);
        processDialog(dialog);
        Optional<String> result=dialog.showAndWait();

        if(result.isPresent()){
            isTranslationMode=araRadio.isSelected();
            generalPrefsJson.put(IS_TRANSLATION_MODE_JSON_KEY,isTranslationMode);
            updateControlsText(); //to translate already existing start window
            initSettingsMenu(); //to translate Settings menu
            generalPrefsJson.put(IS_FIRST_LAUNCH_JSON_KEY,false);
            saveJsonObj(GENERAL_PREFS_FILE_NAME,generalPrefsJson);
            return true;
        }
        else {
           boolean exit= showConfirmationDialog("Quit Jowil Stats", "Are you sure you want to exit?", stage.getOwner());
           if(exit)
               return false;
           else
               return showLanguageFirstTimeDialog(engRadio.isSelected());
        }




    }

    private boolean checkActivationKey() {
        String currentActKey=(String)generalPrefsJson.get(ACTIVATION_KEY_JSON_KEY);

        String correctActKey=getActivationKey();
        if(isDevMode)
            System.out.println(correctActKey);
        if(currentActKey.equals(correctActKey))
            return true;
        else{
            if(showActivationKeyDialog(correctActKey)){
                generalPrefsJson.put(ACTIVATION_KEY_JSON_KEY,correctActKey);
                saveJsonObj(GENERAL_PREFS_FILE_NAME,generalPrefsJson);
                showActivationSuccessDialog();
                return true;
            }
            else
                return false;
        }


    }



    private boolean showActivationKeyDialog(String correctActKey,String... boxesText) {
        Dialog<String> dialog = new Dialog<>();
        dialog.setTitle("Software Activation");
        dialog.setHeaderText("Jowil Stats needs activation");
        dialog.setGraphic(null);


        dialog.getDialogPane().getStyleClass().add("projectsDialog");
        //dialog.getDialogPane().setStyle("-fx-background-color:transparent");
        dialog.setResizable(false);

        double targetHeight=resY*0.5,targetX=resX*0.3,targetY=resY*0.24;

        dialog.heightProperty().addListener(((observable, oldValue, newValue) -> {
            if((double)newValue!=targetHeight)
                dialog.setHeight(targetHeight);
        }));
//        dialog.xProperty().addListener(((observable, oldValue, newValue) -> {
//            if((double)newValue!=targetX)
//                dialog.setX(targetX);
//        }));
//
//        dialog.yProperty().addListener(((observable, oldValue, newValue) -> {
//            if((double)newValue!=targetY)
//                dialog.setY(targetY);
//        }));


        ButtonType activateType=new ButtonType("Activate");
        dialog.getDialogPane().getButtonTypes().addAll(activateType);
        Button activateButton=(Button)dialog.getDialogPane().lookupButton(activateType);
        dialog.getDialogPane().lookup(".header-panel .label").setStyle("-fx-font-size:"+resX*15/1280);


        AnchorPane anchorPane=new AnchorPane();

        Label mainLabel=new Label("-The activation key is 16 characters long and is based on your serial number.\n-You can contact any of our software distributors to get the activation key\n required for your Jowil Stats copy.");
        Label serialKeyLabel=new Label("Serial Number:");
        Label activationLabel=new Label("Activation Key:");


        String volSerial=getVolSerialNumber();
        volSerial=volSerial.substring(0,4)+"-"+volSerial.substring(4);
        Label serialValueLabel=new Label(volSerial);

        double fontSize=resX*12/1280;
        serialValueLabel.setFont(new Font("System Bold",fontSize));
        serialKeyLabel.setFont(new Font(fontSize));
        mainLabel.setFont(new Font(fontSize));
        activationLabel.setFont(new Font(fontSize));
        activationLabel.setPadding(new Insets(3,0,0,0));


        double spacing=resX*4/1280;

        HBox serialHBox=new HBox(serialKeyLabel,serialValueLabel);
        serialHBox.setSpacing(spacing);


        ArrayList<TextField>activationTextFields=new ArrayList<>();
        HBox activationHBox=new HBox(spacing);
        activationHBox.setNodeOrientation(NodeOrientation.LEFT_TO_RIGHT);
        ArrayList<Node> actHBoxChildren=new ArrayList<>();
        if(!isTranslationMode)
            actHBoxChildren.add(activationLabel);
        double textFieldSize=resX*50/1280;

        for(int i=0;i<3;i++) {
            TextField textField=new TextField();
            textField.setPrefWidth(textFieldSize);
            if(boxesText.length>i)
                textField.setText(boxesText[i]);
            activationTextFields.add(textField);
            actHBoxChildren.add(textField);
            Label label=new Label("-");
            label.setPadding(activationLabel.getPadding());
            label.setFont(new Font(fontSize));
            actHBoxChildren.add(label);
        }


        TextField finalTextField=new TextField();
        finalTextField.setPrefWidth(textFieldSize);
        if(boxesText.length>3)
            finalTextField.setText(boxesText[3]);
        activationTextFields.add(finalTextField);
        actHBoxChildren.add(finalTextField);

        if(isTranslationMode)
            actHBoxChildren.add(activationLabel);


        activationHBox.getChildren().setAll(actHBoxChildren);

        Platform.runLater(()->activationTextFields.get(0).requestFocus());

        for(TextField textField:activationTextFields){
            textField.addEventFilter(KeyEvent.KEY_PRESSED, event -> {
                if (event.getCode() == KeyCode.ENTER)
                    activateButton.fire();
            });

            textField.setOnKeyTyped(event -> {
                if(event.getCharacter().charAt(0)==' ')
                    event.consume();
            });

            textField.textProperty().addListener(((observable, oldValue, newValue) -> {
                if(newValue.length()==4) {
                    int nextIndex = activationTextFields.indexOf(textField) + 1;

                    if(nextIndex!=4)
                        activationTextFields.get(nextIndex).requestFocus();
                }
            }));
        }


        mainLabel.setLayoutX(resX*20/1280);
        mainLabel.setLayoutY(resY*20/1280);
        serialHBox.setLayoutX(mainLabel.getLayoutX());
        serialHBox.setLayoutY(resY*155/1280);
        activationHBox.setLayoutX(mainLabel.getLayoutX());
        activationHBox.setLayoutY(resY*230/1280);


        anchorPane.getChildren().addAll(mainLabel,serialHBox,activationHBox);
        dialog.getDialogPane().setPadding(new Insets(5,20,0,20));
        dialog.getDialogPane().setContent(anchorPane);
        processDialog(dialog);

        dialog.setY(realResY*0.26);
        dialog.setX(realResX*0.3);
        dialog.getDialogPane().lookupButton(activateType).setStyle("-fx-border-color: #095c90;-fx-text-fill:#095c90;");
        Optional<String> result=dialog.showAndWait();

        if(!result.isPresent()) {//close was pressed
            boolean exit=showConfirmationDialog("Quit Jowil Stats","Are you sure you want to exit?", stage.getOwner());

            if(exit)
                return false;
            else
                return showActivationKeyDialog(correctActKey,activationTextFields.get(0).getText(),
                        activationTextFields.get(1).getText(),activationTextFields.get(2).getText(),activationTextFields.get(3).getText());

        }


        StringBuilder submittedKey=new StringBuilder();
        StringBuilder submittedKeyDashed=new StringBuilder();




        for(TextField textField:activationTextFields){
                submittedKey.append(textField.getText());
                submittedKeyDashed.append(textField.getText());
                submittedKeyDashed.append('-');

        }


        submittedKeyDashed.deleteCharAt(submittedKeyDashed.length()-1);


        if(submittedKey.toString().toLowerCase().equals(correctActKey))
            return true;
        else{
            showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Wrong Activation Key",
                    constructMessage("The activation key"," \""+submittedKeyDashed+"\" ","is incorrect."));
            return showActivationKeyDialog(correctActKey,activationTextFields.get(0).getText(),
                    activationTextFields.get(1).getText(),activationTextFields.get(2).getText(),activationTextFields.get(3).getText());
        }

    }

    private void showActivationSuccessDialog() {



        Dialog<String> dialog = new Dialog<>();
        dialog.setTitle("Activation Success");


        ImageView pic=new ImageView();
        pic.setImage(new Image("Images/Checked_50px.png"));
        pic.setFitWidth(50);
        pic.setFitHeight(50);

        Label mainLabel=new Label("Jowil Stats activated successfully.");
        mainLabel.setFont(new Font(14));

        HBox hBox=new HBox(5,pic,mainLabel);
        hBox.setAlignment(Pos.CENTER);

        hBox.setMinSize(180,100);


        dialog.getDialogPane().getButtonTypes().setAll(new ButtonType("Start",ButtonBar.ButtonData.OK_DONE));
        dialog.getDialogPane().setContent(hBox);

        processDialog(dialog);
        dialog.showAndWait();


    }

    private static String getActivationKey(){


        long serialNumber;

        serialNumber=Long.parseLong(getVolSerialNumber(),16);

        ArrayList<Character> allChars= new ArrayList<>( );

        for(int i = 48 ; i < 58 ; i++) {
            allChars.add((char)(i)) ; // add numbers
        }


        for(int i = 65 ; i < 91 ; i++) {
            allChars.add((char)(i)) ; // add numbers
        }



        Random gen = new Random(serialNumber) ;

        ArrayList<Character> shuffledChars = (ArrayList)allChars.clone() ;
        Collections.shuffle((ArrayList)shuffledChars,gen) ;
        String output = "";
        for (int i = 1; i < 17; i++) {
            output+=allChars.get(Math.abs(gen.nextInt())%36) ;
        }
        return output.toLowerCase();
    }

    private static String getVolSerialNumber() {

        String result = "";
        try {

            Process p = Runtime.getRuntime().exec("cmd.exe /C vol");
            BufferedReader input =
                    new BufferedReader
                            (new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = input.readLine()) != null) {
                result += line;
            }
            input.close();
        }
        catch(Exception e){
            e.printStackTrace();
        }

        String  [] words=result.split(" ");

        return words[words.length-1].replace("-","");
    }



    private void initDataDirPath() {
        String path = getClass().getProtectionDomain().getCodeSource().getLocation().getPath();
        try {
            isDevMode=path.endsWith("classes/");
            String decodedPath = URLDecoder.decode(isDevMode?path:new File(path).getParentFile().getPath(), "UTF-8"); //the classes check identifies wither the function was called in a deployed jar or normal development classes
            path=decodedPath+"/data/";
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        dataDirPath=path;
    }

    private void loadTranslationsJson() {

        translations=loadJsonObj(TRANSLATIONS_FILE_NAME);

        if(translations==null) { //if failed in loading
            isTranslationMode=false;
            translations = new HashMap<>();
        }

    }

    @Override
    protected void saveChanges() {

    }

    @Override
    protected Controller getNextController() { //not used here
        return null;
    }

    @Override
    protected void buildComponentsGraph(){

        langAnc.getChildren().addAll(langLabel,englishRadio,arabicRadio);
        resScalingAnc.getChildren().addAll(resScalingLabel,resScalingToggle);
        settingsMenu.getItems().addAll(langMenuItem,new SeparatorMenuItem(),resScalingMenuItem);
    }

    @Override
    protected void updateSizes(){


        super.updateSizes();






        double upperHeight=(int)(rootHeight*0.245);
        double lowerHeight=1-upperHeight;

        Font jowilLabelFont=new Font("System Bold",rootWidth*0.059);
        jowilLabel.setFont(jowilLabelFont);

//        upperImageView.setFitWidth(rootWidth);
//        upperImageView.setFitHeight(upperHeight);

        lowerAncPane.setPrefWidth(rootWidth);
        backImageView.setFitWidth(rootWidth);
        backImageView.setFitHeight(rootHeight);
        //lowerAncPane.setPrefHeight(lowerHeight);

        jowilLabel.setLayoutX(0.353*rootWidth);
        jowilLabel.setLayoutY(0.12*rootHeight);


        double vSpacing=rootHeight*0.15;

        logoImageView.setFitWidth(rootWidth*0.058);
        logoImageView.setFitHeight(logoImageView.getFitWidth());
        logoImageView.setLayoutX(0.474*rootWidth);
        //logoImageView.setLayoutY(jowilLabel.getLayoutY()+rootHeight*0.09);
        logoImageView.setLayoutY(jowilLabel.getLayoutY()+vSpacing);


        openStack.setLayoutX(0.25*rootWidth);
        openStack.setLayoutY(logoImageView.getLayoutY()+vSpacing);

        openRect.setWidth(0.15*rootWidth); // kant 0.173
        openRect.setHeight(0.18*rootWidth); //kant 0.4
        openRect.setArcWidth(rootWidth*0.01);
        openRect.setArcHeight(openRect.getArcWidth());

        openImageView.setFitWidth(0.055*rootWidth);
        openImageView.setFitHeight(openImageView.getFitWidth());


        Insets buttonLabelsMargin=new Insets(0.59*openRect.getHeight(),0,0,0);
        Insets buttonImagesMargin=new Insets(0,0,0.118*openRect.getHeight(),0);


        StackPane.setMargin(openLabel,buttonLabelsMargin);
        StackPane.setMargin(openImageView,buttonImagesMargin);


        newStack.setLayoutX(0.595*rootWidth);
        newStack.setLayoutY(openStack.getLayoutY());

        newRect.setWidth(openRect.getWidth());
        newRect.setHeight(openRect.getHeight());
        newRect.setArcWidth(openRect.getArcWidth());
        newRect.setArcHeight(openRect.getArcHeight());

        newImageView.setFitWidth(openImageView.getFitWidth());
        newImageView.setFitHeight(openImageView.getFitHeight());

        StackPane.setMargin(newLabel,buttonLabelsMargin);
        StackPane.setMargin(newImageView,buttonImagesMargin);


    }


    private void initSettingsMenu() {

        double flagsSize=15;
        langAnc.setPrefWidth(langAncWidth);
        langLabel.setFont(new Font("System Bold",resX*13/1280));
        englishRadio.setToggleGroup(langToggleGroup);
        ImageView englishImageView=new ImageView("Images/USA.png");
        englishImageView.setFitWidth(flagsSize);
        englishImageView.setFitHeight(flagsSize);
        englishRadio.setGraphic(englishImageView);
        arabicRadio.setToggleGroup(langToggleGroup);
        ImageView arabicImageView=new ImageView("Images/Egypt.png");
        arabicImageView.setFitWidth(flagsSize);
        arabicImageView.setFitHeight(flagsSize);
        arabicRadio.setGraphic(arabicImageView);
        englishRadio.setLayoutY(27);
        arabicRadio.setLayoutY(englishRadio.getLayoutY());
        AnchorPane.setLeftAnchor(englishRadio,5.0);
        AnchorPane.setRightAnchor(arabicRadio,2.0);
        if(isTranslationMode)
            arabicRadio.setSelected(true);
        else
            englishRadio.setSelected(true);
        arabicRadio.setPadding(Insets.EMPTY);
        langToggleGroup.selectedToggleProperty().addListener((observable, oldValue, newValue) -> {

            if(isTranslationMode && newValue==arabicRadio || !isTranslationMode && newValue==englishRadio) //the selection wasn't made by user
                return;

            //the string will be translated in the opposite language option so in arabic string it will be change to english
            boolean isChange=showConfirmationDialog("Change Language","Are you sure you want to change the program language to Arabic?",
                    stage.getOwner());

            if(isChange){
                isTranslationMode=!isTranslationMode;
                if(generalPrefsJson!=null)
                    generalPrefsJson.put(IS_TRANSLATION_MODE_JSON_KEY,isTranslationMode);
                saveJsonObj(GENERAL_PREFS_FILE_NAME,generalPrefsJson);
                stage.close();
                new StartController().startWindow();
            }
            else
                langToggleGroup.selectToggle(oldValue);

        });
        langMenuItem.getStyleClass().add("nonSelectableMenuItem");
        langMenuItem.setHideOnClick(false);



        resScalingLabel.setFont(langLabel.getFont());
        resScalingLabel.setLayoutY(5);
        resScalingToggle.setText(isNormalScalingMode?offString:onString);
        resScalingToggle.setSelected(!isNormalScalingMode);
        resScalingToggle.selectedProperty().addListener((observable, oldValue, newValue) -> {

            resScalingToggle.setText(newValue ? onString : offString);
            if(isNormalScalingMode!=newValue) //the selection wasn't made by user
                return;
            String option=newValue?"enable":"disable";
            boolean isChange=showConfirmationDialog("Relative Resolution Scaling","Are you sure you want to "+option+" relative resolution scaling?",
                    stage.getOwner());

            if(isChange){
                isNormalScalingMode=!isNormalScalingMode;
                if(generalPrefsJson!=null)
                    generalPrefsJson.put(IS_NORMAL_SCALING_MODE_JSON_KEY,isNormalScalingMode);
                saveJsonObj(GENERAL_PREFS_FILE_NAME,generalPrefsJson);
                stage.close();
                new StartController().startWindow();
            }
            else
                resScalingToggle.setSelected(oldValue);

        });
        resScalingToggle.setLayoutY(17);
        resScalingToggle.setStyle("-jfx-toggle-color:#095c90;");
        resScalingToggle.setPadding(Insets.EMPTY);
        resScalingMenuItem.getStyleClass().add("nonSelectableMenuItem");
        resScalingMenuItem.setHideOnClick(false);




        settingsMenu.setOnShown(event->translateAllNodes(settingsMenu.getScene().getRoot()));
        if(isTranslationMode)
            settingsMenu.getScene().getRoot().setNodeOrientation(NodeOrientation.RIGHT_TO_LEFT);



    }


    private void showNewProjectNameDialog(String initialText){

        String defaultText=isTranslationMode&& translations.containsKey("New Project")?translations.get("New Project"):"New Project";
        Dialog<String> dialog = new Dialog<>();
        dialog.setTitle("Set Project Name");

        ImageView pic=new ImageView();
        pic.setImage(new Image("Images/Add Folder_96px.png"));
        dialog.setGraphic(pic);
        pic.setFitWidth(resX*30/1280);
        pic.setFitHeight(resX*30/1280);

        VBox contentVBox=new VBox(resX*12/1280);

        HBox textHBox=new HBox(resX*7/1280);
        textHBox.setPadding(new Insets(resX*4/1280,0,0,0));
        TextField projNameTextField = new TextField();
        projNameTextField.setText(initialText.isEmpty()?defaultText:initialText);
        Label label=new Label(isTranslationMode && translations.containsKey("Project Name:")?translations.get("Project Name:"):"Project Name:");
        label.prefHeightProperty().bind(projNameTextField.heightProperty());
        label.setAlignment(Pos.CENTER);
        textHBox.getChildren().addAll(label, projNameTextField);


        double radioIconsSize=resX*18/1280;
        HBox radiosHBox=new HBox(resX*35/1280);
        radiosHBox.setPadding(new Insets(0,0,0,resX*12/1280));
        ToggleGroup toggleGroup= new ToggleGroup();
        RadioButton testRadio=new RadioButton("Test");
        ImageView testImageView=new ImageView("Images/Pass Fail_48px.png");
        testImageView.setFitWidth(radioIconsSize);
        testImageView.setFitHeight(radioIconsSize);
        testRadio.setGraphic(testImageView);
        RadioButton questRadio=new RadioButton("Questionnaire");
        ImageView questImageView=new ImageView("Images/Questionnaire_48px.png");
        questImageView.setFitWidth(radioIconsSize);
        questImageView.setFitHeight(radioIconsSize);
        questRadio.setGraphic(questImageView);
        testRadio.setToggleGroup(toggleGroup);
        questRadio.setToggleGroup(toggleGroup);
        toggleGroup.selectedToggleProperty().addListener((observable, oldValue, newValue) -> {
            generalPrefsJson.put(IS_QUEST_MODE_JSON_KEY,newValue==questRadio);
        });
        if((Boolean)generalPrefsJson.get(IS_QUEST_MODE_JSON_KEY))
            questRadio.setSelected(true);
        else
            testRadio.setSelected(true);




        radiosHBox.getChildren().addAll(testRadio,questRadio);

        contentVBox.setAlignment(Pos.BOTTOM_LEFT);

        Pane emptyPane=new Pane();
        emptyPane.setBackground(Background.EMPTY);
        emptyPane.setPrefHeight(0);
        emptyPane.setPrefWidth(10);
        JFXCheckBox translateFormCheckBox=new JFXCheckBox("Translate form content to arabic",resX*15/1280);
        translateFormCheckBox.setPadding(new Insets(resX*12/1280,0,0,resX*12/1280));
        translateFormCheckBox.setSelected((Boolean)generalPrefsJson.get(IS_TRANSLATE_FORM_CONTENT_JSON_KEY));
        translateFormCheckBox.selectedProperty().addListener(((observable, oldValue, newValue) -> {
            generalPrefsJson.put(IS_TRANSLATE_FORM_CONTENT_JSON_KEY,newValue);
        }));

        //"Answer key in first row of students responses"
        JFXCheckBox firstRowAnswerKeyCheckBox=new JFXCheckBox("Use a separate file for answer key",resX*15/1280);
        firstRowAnswerKeyCheckBox.setPadding(translateFormCheckBox.getPadding());
        firstRowAnswerKeyCheckBox.setSelected(!(Boolean)generalPrefsJson.get(IS_ANSWER_KEY_IN_FIRST_ROW_JSON_KEY));
        firstRowAnswerKeyCheckBox.selectedProperty().addListener(((observable, oldValue, newValue) -> {
            generalPrefsJson.put(IS_ANSWER_KEY_IN_FIRST_ROW_JSON_KEY,!newValue);
        }));


        VBox checkBoxesVBox=new VBox(resY*5/1280,translateFormCheckBox,firstRowAnswerKeyCheckBox);

        contentVBox.getChildren().addAll(textHBox,radiosHBox,checkBoxesVBox);


        dialog.getDialogPane().setContent(contentVBox);
        dialog.getDialogPane().setStyle("-fx-font-size:"+resX*12/1280);

        dialog.getDialogPane().getButtonTypes().setAll(ButtonType.OK,ButtonType.CANCEL);

        dialog.getDialogPane().lookupButton(ButtonType.OK).setStyle("-fx-border-color: #095c90;-fx-text-fill:#095c90;");

        dialog.setResultConverter(dialogButton -> {

            if (dialogButton == ButtonType.OK)
                return projNameTextField.getText();

            return null;

        });


        processDialog(dialog);
        Platform.runLater(()->projNameTextField.requestFocus());
        dialog.setX(realResX*0.38);
        dialog.setY(realResY*0.28);
        Optional<String> result = dialog.showAndWait();


        if (result.isPresent()){
            String projName=result.get();
            if(projName.trim().isEmpty()){
                showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Project Name Error","Project name cannot be empty.");
                showNewProjectNameDialog(projName);
            }
            else if(isProjectExists(projName)) {
                showAlertAndWait(Alert.AlertType.ERROR, stage.getOwner(), "Project Name Error", "" +
                        constructMessage("A project with the name"," \"" + projName + "\" ","already exists."));
                showNewProjectNameDialog(projName);
            }

            else if(isProjectNameInvalid(projName)){
                showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Project Name Error","Invalid project name. Project Name cannot" +
                        " contain any of the following characters: "+"\n< > : \" / \\ | ? *");
                showNewProjectNameDialog(projName);
            }
            else {
                Controller.projectName=projName;
                Controller.isOpenMode=false;
                Controller.currentOpenedProjectJson=null;
                Controller.isQuestMode=questRadio.isSelected();
                Controller.isTranslateFormContent=translateFormCheckBox.isSelected();
                Controller.isAnswerKeyInFirstRow=!firstRowAnswerKeyCheckBox.isSelected();
                new FileConfigController().startWindow();
            }
        }

    }

    private boolean isProjectExists(String s) {
        loadProjectNames();
        return projectsNames.contains(s);
    }

    private boolean isProjectNameInvalid(String name){
        Pattern pattern = Pattern.compile("[<>:\"/\\\\|?*]");
        return pattern.matcher(name).find();
    }

    private void loadProjectsJson(){

        savedProjectsJson=loadJsonObj(SAVED_PROJECTS_FILE_NAME);
        if(savedProjectsJson==null){
            if(showJsonError())
                new StartController().startWindow();

            stage.close();
            return;
        }
    }

    private boolean showJsonError() {
        Alert alert = new Alert(Alert.AlertType.ERROR);

        alert.setTitle("Projects Loading Error");
        alert.getButtonTypes().add(ButtonType.CANCEL);
        alert.setHeaderText(null);
        alert.setContentText("Cannot load existing projects. Would you like to restart Jowil Stats?");
        processDialog(alert);
        Optional<ButtonType> result=alert.showAndWait();

        return result.isPresent() && result.get()==ButtonType.OK;
    }

    private void  showExistingProjects(){
        Dialog<String> dialog = new Dialog<>();
        dialog.setTitle("Existing Projects");
        dialog.setHeaderText("Projects");
        dialog.setGraphic(null);
        dialog.getDialogPane().setMinWidth(resX*300/1280);

        dialog.getDialogPane().getStyleClass().add("projectsDialog");
        //dialog.getDialogPane().setStyle("-fx-background-color:transparent");
        dialog.setResizable(true);
        //dialog.getDialogPane().setMinSize(resX*0.5,resY*0.5);
        dialog.setWidth(Math.min(resX*0.6,Screen.getPrimary().getVisualBounds().getWidth()*0.85));
        dialog.setHeight(resY*0.8);

        dialog.getDialogPane().getButtonTypes().setAll(ButtonType.CLOSE);

        Button butt=(Button)dialog.getDialogPane().lookupButton(ButtonType.CLOSE); //butt is needed to be able to close dialog using x
        butt.setVisible(false);
        dialog.getDialogPane().lookup(".header-panel .label").setStyle("-fx-font-size:"+resX*15/1280);

        JFXListView projectsList=new JFXListView();



        existingProjectsListItems=FXCollections.observableArrayList();
        loadProjectNames();


        projectsList.setCellFactory(event->OpenProjectCell.createOpenProjectCell(dialog,this));


        for(String name:projectsNames){

            existingProjectsListItems.add(name);
            System.out.println(name);
        }


        Label emptyLabel=new Label(isTranslationMode && translations.containsKey("No Projects Exist")?translations.get("No Projects Exist"):"No Projects Exist");
        emptyLabel.setFont(new Font("System Bold",15));
        projectsList.setPlaceholder(emptyLabel);



        if(existingProjectsListItems.isEmpty())
            projectsList.setPrefHeight(0); //prevents enlargement of listView

        projectsList.setPrefWidth(dialog.getWidth());


        dialog.getDialogPane().setPadding(new Insets(20,20,0,20));
        projectsList.setItems(existingProjectsListItems);

        dialog.getDialogPane().setContent(projectsList);
        processDialog(dialog);
        dialog.showAndWait();

    }


    /*
    used to refresh project names 3shn if it was changed anywhere in the app.
     */
    void loadProjectNames(){
        JSONArray projectsArr=(JSONArray)savedProjectsJson.get("projects");
        projectsNames=new ArrayList<>();

        for(int i=0;i<projectsArr.size();i++){
            JSONObject project=(JSONObject)projectsArr.get(i);
            projectsNames.add((String)project.get("name"));
        }
    }

    public void startOpenMode(String projName,Dialog dialog) {

        Controller.projectName = projName;
        Controller.isOpenMode=true;
        ManualModeController.isIgnoreSavedObjectiveWeights=false;

        int projIndex=projectsNames.indexOf(projName);
        JSONArray projects=(JSONArray)savedProjectsJson.get("projects");
        currentOpenedProjectJson= (JSONObject)projects.get(projIndex);
        Controller.isQuestMode=(Boolean)currentOpenedProjectJson.get(IS_QUEST_MODE_JSON_KEY);
        Controller.isTranslateFormContent=(Boolean)currentOpenedProjectJson.get(IS_TRANSLATE_FORM_CONTENT_JSON_KEY);
        Controller.isAnswerKeyInFirstRow=(Boolean)currentOpenedProjectJson.get(IS_ANSWER_KEY_IN_FIRST_ROW_JSON_KEY);
        try {
            Thread.sleep(250);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("proj: " + Controller.projectName);
        dialog.close();
        new FileConfigController().startWindow();
    }


    public void deleteProject(String projName) {


        if(!showConfirmationDialog("Delete Project", constructMessage("Are you sure you want to delete project"," \""+projName+"\" ","?"),stage.getOwner()))
            return;
        JSONArray projects=(JSONArray)savedProjectsJson.get("projects");
        for(int i=0;i<projects.size();i++){
            if(((JSONObject)projects.get(i)).get(PROJECT_NAME_JSON_KEY).equals(projName)){
                projects.remove(i);
                break;
            }
        }
        existingProjectsListItems.remove(projName);
        projectsNames.remove(projName);
        saveJsonObj(SAVED_PROJECTS_FILE_NAME,savedProjectsJson);



    }

    public void openProjectInExplorer(String projName) {

        int projIndex=projectsNames.indexOf(projName);
        JSONArray projects=(JSONArray)savedProjectsJson.get("projects");
        JSONObject proj=(JSONObject)projects.get(projIndex);

        String dirPath=(String)proj.get(Controller.REPORTS_OUT_PATH_JSON_KEY)+"\\"+projName;


        File file = new File (dirPath);
        Desktop desktop = Desktop.getDesktop();


        try {
            desktop.open(file);
        } catch (IOException | IllegalArgumentException e) {
            showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Directory Error","Cannot open the project in explorer. Make sure that the project folder exists.");
        }


    }
}
