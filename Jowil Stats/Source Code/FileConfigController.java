package Jowil;

import com.jfoenix.controls.*;
import com.jfoenix.validation.base.ValidatorBase;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIconView;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Popup;

import java.io.*;
import java.util.ArrayList;
import java.util.Optional;
import java.util.regex.Pattern;

import javafx.util.Callback;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import static Jowil.CSVHandler.NOT_AVAILABLE;


public class FileConfigController extends Controller {

    //components

    @FXML
    private JFXTextField mainFileTextField;
    @FXML
    private StackPane mainFileChooser;

    @FXML
    private ImageView mainChooserButtonImage;

    @FXML
    private JFXButton nextButton;

    @FXML
    private JFXTextField answersFileTextField;

    @FXML
    private JFXComboBox identifierCombo;

    @FXML
    private JFXComboBox formCombo;

    @FXML
    private JFXToggleButton manualModeToggle;


    @FXML
    private StackPane answersFileChooser;

    @FXML
    private HBox mainHBox;

    @FXML
	private ImageView answersChooserButtonImage;
	@FXML
    private HBox answersHBox;

    @FXML
    private HBox combosHBox;


    @FXML
    private AnchorPane combosAnchor;

    @FXML
    private VBox contentVbox;


    @FXML
    FontAwesomeIconView mainChooserIcon;


    @FXML
    FontAwesomeIconView answersChooserIcon;




    VBox subjVBox = new VBox();
    JFXSlider slider = new JFXSlider();
    final Popup popup = new Popup();


    //data fields
    private String lastDir;
    File csvFile;
    private static final String FX_LABEL_FLOAT_TRUE = "-fx-label-float:true;";
    private static final String EM1 = "1em";
    private static final String ERROR = "error";
    private ArrayList<String> filteredInfoHeaders;
    private ObservableList<String> combosItems = FXCollections.observableArrayList();
    private int identifierComboSelectedIndex; //including none at index zero
    private int formComboSelectedIndex; //including none at index zero
    private int mainTextFieldResult = CSVFileValidator.ERROR, answersTextFieldResult = CSVFileValidator.ERROR;
    private String mainTextFieldMessage = "", answersTextFieldMessage = "";
    private boolean isComplexIDAdded = false;
    private int complexIDSize = 0;
    private int complexIdStartIndex;
    private final static int SKIPROW = 0, CONTINUE = 1, CANCEL = 2, DECLARESUBJ = 3;
    final static String NONE_OPTION="None";


    int manualColsCounter = 0;
    int manualIDIndex;
    int manualFormIndex;
    boolean isMainTextFieldValidated;


    //getters and setters


    //Main methods
    FileConfigController() {


        super("FileConfig.fxml", "File configuration", 1.6, 1.45, true, null,"1.png",0,"Files Configuration",resX*700/1280, Integer.MAX_VALUE);


    }


    protected void updateSizes() {
        super.updateSizes();

        contentVbox.setLayoutX(rootWidthToPixels(0.072));
        contentVbox.setLayoutY(rootHeightToPixels(0.13));
        contentVbox.setSpacing(rootHeightToPixels(0.115));
        contentVbox.setAlignment(Pos.BASELINE_LEFT);

        mainHBox.setSpacing(resXToPixels(0.005));
        answersHBox.setSpacing(resXToPixels(0.005));


        mainFileTextField.setPrefWidth(rootWidthToPixels(0.5));
        mainFileTextField.setPrefHeight(resYToPixels(0.04));
        mainFileTextField.setPadding(Insets.EMPTY);


        answersFileTextField.setPrefWidth(rootWidthToPixels(0.5));
        answersFileTextField.setPrefHeight(resYToPixels(0.04));
        answersFileTextField.setPadding(Insets.EMPTY);

        answersChooserIcon.setSize(Double.toString(DEFAULT_FONT_AWESOME_ICON_SIZE));
        mainChooserIcon.setSize(Double.toString(DEFAULT_FONT_AWESOME_ICON_SIZE));


        formCombo.setPadding(Insets.EMPTY);
        formCombo.setPrefWidth(rootWidthToPixels(0.227));
        identifierCombo.setPrefWidth(rootWidthToPixels(0.227));
        manualModeToggle.setPadding(new Insets(rootHeightToPixels(0.03), 0, 0, 0));
        manualModeToggle.setSize(resX*10/1280);


        nextButton.setPrefWidth(resXToPixels(0.07));
        nextButton.setPrefHeight(resXToPixels(0.004));
        nextButton.setLayoutX(rootWidthToPixels(0.78));
        nextButton.setLayoutY(rootHeightToPixels(0.77));





        combosAnchor.setPrefWidth(rootWidthToPixels(0.4));
        AnchorPane.setLeftAnchor(identifierCombo, 0.0);
        AnchorPane.setRightAnchor(formCombo, 0.0);
        combosAnchor.setPadding(new Insets(0, answersFileChooser.getLayoutBounds().getWidth() + answersHBox.getSpacing(), 0, 0));

        subjVBox.setLayoutX(rootWidthToPixels(0.11));
        subjVBox.setLayoutY(rootHeightToPixels(0.5));

    }

    protected void initComponents() {
        initAnswersFileChooser();
        initMainFileChooser();
        initNextButton();
        initMainFileTextField();
        initAnswersFileTextField();
        initIdentifierCombo();
        initFormCombo();
        initManualModeToggle();
        isMainTextFieldValidated=false;

        if(generalPrefsJson!=null)
            lastDir=(String)generalPrefsJson.get(LAST_CSV_DIR_JSON_KEY);


        if (isOpenMode) {

            mainFileTextField.setText((String) currentOpenedProjectJson.get(RESPONSES_FILE_PATH_JSON_KEY));
            String answersFilePath=(String) currentOpenedProjectJson.get(ANSWERS_FILE_PATH_JSON_KEY);
            answersFileTextField.setText(isAnswerKeyInFirstRow?"":answersFilePath);
            mainFileTextField.setEditable(false);
            answersFileTextField.setEditable(false);
            CSVHandler.setResponsesFilePath(mainFileTextField.getText()); //because the csv loading functions are not called in openMode so these strings will be equal to null.
            CSVHandler.setAnswerKeyFilePath(answersFilePath); //same as above
            validateMainTextField();
            validateAnswersTextField();
            answersFileChooser.setOpacity(0.3);
            mainFileChooser.setOpacity(0.3);
            answersFileChooser.addEventFilter(MouseEvent.ANY,event -> event.consume()); //block all mouse events on choosers in open mode
            mainFileChooser.addEventFilter(MouseEvent.ANY,event -> event.consume());

        }
        if(isQuestMode) {
            answersFileTextField.setDisable(true);
            answersFileChooser.setOpacity(0.3);
            answersFileChooser.addEventFilter(MouseEvent.ANY,event -> event.consume());
            identifierCombo.setDisable(true);
            identifierCombo.disabledProperty().addListener(((observable, oldValue, newValue) -> {
                if(!newValue)
                    identifierCombo.setDisable(true);
            }));
            formCombo.setDisable(true);
            formCombo.disabledProperty().addListener(((observable, oldValue, newValue) -> {
                if(!newValue)
                    formCombo.setDisable(true);
            }));
        }

        if(isAnswerKeyInFirstRow) {
            answersFileTextField.setDisable(true);
            answersFileChooser.setOpacity(0.3);
            answersFileChooser.addEventFilter(MouseEvent.ANY,event -> event.consume());
        }


        if(!isOpenMode && isDevMode){
            mainFileTextField.setText(".\\src\\test\\AppTestCSVs\\TestGOnly.csv");
            if(!isQuestMode && !isAnswerKeyInFirstRow)
                answersFileTextField.setText(".\\src\\test\\AppTestCSVs\\alexAnswerKeysGOnlyWithEndProgressiveBlanks.csv");
        }





    }

    @Override
    protected Controller getNextController() {
        return null;
    }

    @Override
    protected void saveChanges() {

        selectedIdentifierName = (String) identifierCombo.getSelectionModel().getSelectedItem();
        selectedFormColName = CSVHandler.getFormsCount()==1?NONE_OPTION:(String) formCombo.getSelectionModel().getSelectedItem();
        saveIdentifierColumn();
        saveFormColumn();


    }


    @Override
    protected void buildComponentsGraph() {

//        subjVBox.getChildren().add(toggleButton);
//        subjVBox.getChildren().add(slider);
//        rootPane.getChildren().add(subjVBox);

        //super.buildComponentsGraph();
        Label label = new Label();
        label.setText("This is an error message");
        label.getStyleClass().add("chat-bubble");
        popup.getContent().add(label);

    }

    //helper methods
    protected void initNextButton() {

//
        nextButton.setOnMouseClicked(t -> {


            rootPane.requestFocus();


            if (!isMainTextFieldValidated)
                validateMainTextField();

            boolean isManualMode = manualModeToggle.isSelected();


            validateAnswersTextField();


            if (mainTextFieldResult == CSVFileValidator.WARNING)
                isManualMode = true;

            //check for errors
            if (mainFileTextField.getText().length() == 0) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error", "No students responses file provided.");
                return;
            }

            if (answersFileTextField.getText().length() == 0 && !isQuestMode && !isAnswerKeyInFirstRow) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Answer Key File Error", "No answer key file provided.");
                return;
            }

            if (mainTextFieldResult == CSVFileValidator.ERROR) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error", constructMessage("Error in students responses file: ", mainTextFieldMessage));
                return;
            }

            if (answersTextFieldResult == CSVFileValidator.ERROR && !isQuestMode && !isAnswerKeyInFirstRow) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Answer Key File Error", constructMessage("Error in answer key file: ", answersTextFieldMessage));
                return;
            }



            if(isAnswerKeyInFirstRow)
                CSVHandler.setFormsCount(1);


            int formsCount = isOpenMode ? ((JSONArray) currentOpenedProjectJson.get(OBJ_WEIGHTS_JSON_KEY)).size() : CSVHandler.getFormsCount();


            if (formsCount > 1 && formComboSelectedIndex == 0 && !isManualMode && !isQuestMode) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Answer Key File Error",
                        constructMessage(formsCount + "", " answer keys detected. Form column cannot have a \"None\" value. Select a valid form column to continue."));
                return;
            }

            if(!isQuestMode) {
                if (formsCount == 0) {
                    showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Answer Key File Error",
                            "Answer key file must contain the answers for at least one form.");
                    return;
                }



                int responsesColCount = CSVHandler.getResponsesColsCount();
                int answersColCount = CSVHandler.getAnswersColsCount();

                if (responsesColCount != answersColCount && !isOpenMode && !isAnswerKeyInFirstRow) {
                    showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Columns Count Mismatch", constructMessage("Student responses file contains ",
                            responsesColCount + "", " columns, while the answer key file contains ", answersColCount + "", " columns."));
                    return;
                }
            }

            if(generalPrefsJson!=null) {
                generalPrefsJson.put(LAST_CSV_DIR_JSON_KEY, lastDir);
                saveJsonObj(GENERAL_PREFS_FILE_NAME, generalPrefsJson);
            }


            ManualModeController.setIsManualModeUsedBefore(false);


            if (isOpenMode) {
                CSVHandler.setIsResponsesContainsHeaders((boolean) currentOpenedProjectJson.get(IS_RESPONSES_CONTAINS_HEADERS_JSON_KEY));
                CSVHandler.setIsAnswerKeyContainsHeaders((boolean)currentOpenedProjectJson.get(IS_ANSWER_KEY_CONTAINS_HEADERS_JSON_KEY));
                CSVHandler.setSavedAnswerKeyCSV((ArrayList<ArrayList<String>>)currentOpenedProjectJson.get(SAVED_ANSWER_KEY_CSV_JSON_KEY));
                CSVHandler.setSavedResponsesCSV((ArrayList<ArrayList<String>>)currentOpenedProjectJson.get(SAVED_RESPONSES_CSV_JSON_KEY));
                CSVHandler.setFormsCount(Integer.parseInt((String)currentOpenedProjectJson.get(FORMS_COUNT_JSON_KEY)));
                CSVHandler.setSubjStartIndex(Integer.parseInt((String) currentOpenedProjectJson.get(SUBJ_COL_START_INDEX_JSON_KEY)));
                CSVHandler.setSubjEndIndex(Integer.parseInt((String)currentOpenedProjectJson.get(SUBJ_COL_END_INDEX_JSON_KEY)));
                CSVHandler.setSubjQuestionsCount(Integer.parseInt((String)currentOpenedProjectJson.get(SUBJ_Q_COUNT_JSON_KEY)));
                CSVHandler.setQuestionsColStartIndex(Integer.parseInt((String)currentOpenedProjectJson.get(Q_COL_START_INDEX_JSON_KEY)));
                CSVHandler.setQuestionsColEndIndex(Integer.parseInt((String)currentOpenedProjectJson.get(Q_COL_END_INDEX_JSON_KEY)));
                loadSavedProjectJson();
                if(isManualMode){
                    openManualMode();
                }
                else {

                    saveChanges();
                    CSVHandler.loadSavedAnswerKey();
                    try {
                        CSVHandler.loadSavedCSV();
                    } catch (CSVHandler.InvalidFormNumberException e) {
                        showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                                constructMessage("Error in students responses file: " , e.getMessage(),". Make sure that you have selected the form column correctly."));
                        return;
                    } catch (CSVHandler.InvalidSubjColumnException e) { //won't happen because subjective columns were already validated when was first project created. ->ignore translation
                        showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                                "Error in students responses file: " + e.getMessage());
                        return;
                    }

                    new GroupsController(this).startWindow();
                    stage.close();
                }
                return;
            }


            if(answersTextFieldResult==CSVFileValidator.WARNING){

                int selectedAction = showHeadersWarningDialog("answer key");
                if (selectedAction == CANCEL)
                    return;

                CSVHandler.setIsAnswerKeyContainsHeaders(selectedAction == SKIPROW);
                if(selectedAction==SKIPROW && formsCount==1){ //the forms count includes the headers row
                    showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Answer Key File Error",
                            "Answer key file must contain the answers for at least one form.");
                    return;
                }

            }
            else
                CSVHandler.setIsAnswerKeyContainsHeaders(true);



            if (isManualMode) {
                //if no headers in responses

                if (mainTextFieldResult == CSVFileValidator.WARNING) {

                    int selectedAction = showHeadersWarningDialog("Students Responses");
                    if (selectedAction == CANCEL)
                        return;
                    CSVHandler.setIsResponsesContainsHeaders(selectedAction == SKIPROW);
                } else //manual mode set by the user && not open mode
                    CSVHandler.setIsResponsesContainsHeaders(true);

                openManualMode();
                return;
            }


            //auto mode->contains headers->ignore headers if switched later to manual mode
            CSVHandler.setIsResponsesContainsHeaders(true);


            if (answersTextFieldResult == CSVFileValidator.SUCCESS && !isAnswerKeyInFirstRow) { //both files contain headers -> check for headers mismatch
                boolean isMismatched;
                try {
                    isMismatched = CSVHandler.isFilesHeadersMismatched(new File(mainFileTextField.getText()), new File(answersFileTextField.getText()));
                } catch (IOException e) {
                    showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                            "Error in reloading files.");
                    return;
                }
                if (isMismatched) {
                    showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Headers Mismatch Error", "Headers of " +
                            "answers and student responses files are not identical.");
                    return;
                }
            }


            try {
                CSVHandler.loadAnswerKeys(isAnswerKeyInFirstRow?CSVHandler.getAnswerKeyFilePath():answersFileTextField.getText(), true);
            } catch (IOException e) {

            } catch (CSVHandler.IllFormedCSVException e) {
                e.printStackTrace(); //msh hat7sl hena la2en lw feh aslun kan hasal f el call el f CSVFileValidator
            } catch (CSVHandler.InConsistentAnswerKeyException e) {
                e.printStackTrace(); //nafs el kalam zy el fo2eha
            }


            if (CSVHandler.isIsAnswerKeyContainsBlanks()) { //questions counts matching in auto mode but correct answers contain blanks
                try {

                    CSVHandler.processHeaders(true); //clean the blanks
                } catch (Exception e) {
                    showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                            "Error in reloading responses file.");
                    return;
                }
            }

            saveChanges();


            try {
                CSVHandler.loadCsv(true);
            } catch (CSVHandler.IllFormedCSVException e) {
                String message=constructMessage("Error in students responses file at row " , e.getRowNumber()+"" ,
                        ". File must contain the same number of columns in all rows.",(e.getRowNumber()==2?" Make sure that the CSV headers have no commas.":""));
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",message);

                return;
            } catch (IOException e) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                        "Error in reading students responses file.");
                return;
            } catch (CSVHandler.InvalidFormNumberException | CSVHandler.InvalidSubjColumnException e) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                        constructMessage("Error in students responses file: " , e.getMessage()));
                return;
            }


            new GroupsController(this).startWindow();
            stage.close();

        });


    }

    private void generateAnswerKeyCSVFromFirstLine(boolean isHeadersExist) {

        if(!isAnswerKeyInFirstRow)
            return;

        BufferedReader input=null;
        try {
             input= new BufferedReader(new FileReader(mainFileTextField.getText()));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        BufferedWriter pw = null;

        String answerKeyFilePath=dataDirPath+"GeneratedAnswerKey.csv";

        try {
            pw =  new BufferedWriter(new FileWriter(answerKeyFilePath));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


        for(int i=0;i<(isHeadersExist?2:1);i++){
            try {
                if(i==1)
                    pw.newLine();
                pw.write(input.readLine());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        try {
            pw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        CSVHandler.setAnswerKeyFilePath(answerKeyFilePath);


    }


    private void loadSavedProjectJson() {
        loadObjectiveGroups();
        loadQNames();
        loadQuestionsChoices();
        loadObjWeights();
        loadSubjWeights();
    }


    private void loadObjectiveGroups() {
        JSONArray jsonGroups = (JSONArray) currentOpenedProjectJson.get(OBJ_GROUPS_JSON_KEY);
        ArrayList<Group> groups = new ArrayList<>();
        for (Object group : jsonGroups) {
            String name = (String) ((JSONObject) group).get("name");
            String qCount = (String) ((JSONObject) group).get("qCount");
            String realQCount= (String) ((JSONObject) group).get("realQCount");
            groups.add(new Group(name, Integer.parseInt(qCount),Integer.parseInt(realQCount)));
        }
        CSVHandler.setDetectedGroups(groups);
    }

    private void loadQNames() {
        CSVHandler.setDetectedQHeaders((JSONArray) currentOpenedProjectJson.get(Q_NAMES_JSON_KEY));
    }

    private void loadQuestionsChoices() {
        Statistics.setQuestionsChoices((JSONArray) currentOpenedProjectJson.get(Q_CHOICES_JSON_KEY));
    }

    private void loadObjWeights() {
        Statistics.setQuestionWeights((JSONArray) currentOpenedProjectJson.get(OBJ_WEIGHTS_JSON_KEY));
    }


    private void loadSubjWeights() {
        Statistics.setSubjMaxScores((JSONArray) currentOpenedProjectJson.get(SUBJ_WEIGHTS_JSON_KEY));
    }


    private void initMainFileChooser() {


        TranslatableTooltip tooltip = new TranslatableTooltip("Open CSV file");

        Tooltip.install(mainFileChooser, tooltip);

        mainFileChooser.setOnMouseClicked(new EventHandler<MouseEvent>
                () {
            public void handle(MouseEvent t) {
                mainFileChooser.setStyle("-fx-background-color:transparent;");
                FileChooser fileChooser = new FileChooser();
                fileChooser.setTitle("Open CSV file");
                fileChooser.setInitialDirectory(new File((lastDir == null ? System.getProperty("user.home") : lastDir)));
                fileChooser.getExtensionFilters().addAll(new FileChooser.ExtensionFilter("CSV", "*.csv"));
                csvFile = fileChooser.showOpenDialog(stage);
                if (csvFile != null) {
                    lastDir = csvFile.getParent();
                    mainFileTextField.setText(csvFile.getPath());
                    validateMainTextField();
                    mainFileTextField.requestFocus();
                    mainFileTextField.deselect();

                }
            }
        });


    }

    private void initAnswersFileChooser() {

        TranslatableTooltip tooltip = new TranslatableTooltip("Open CSV file");
        Tooltip.install(answersFileChooser, tooltip);


        answersFileChooser.setOnMouseClicked(new EventHandler<MouseEvent>
                () {
            public void handle(MouseEvent t) {
                //answersFileChooser.setStyle("-fx-background-color:transparent;");
                FileChooser fileChooser = new FileChooser();
                fileChooser.setTitle("Open CSV file");
                fileChooser.setInitialDirectory(new File((lastDir == null ? System.getProperty("user.home") : lastDir)));
                fileChooser.getExtensionFilters().addAll(new FileChooser.ExtensionFilter("CSV", "*.csv"));
                csvFile = fileChooser.showOpenDialog(stage);
                if (csvFile != null) {
                    lastDir = csvFile.getParent();
                    answersFileTextField.setText(csvFile.getPath());
                    answersFileTextField.requestFocus();
                    answersFileTextField.deselect();
                    validateAnswersTextField();
                }


            }
        });


    }


    private void initMainFileTextField() {


        mainFileTextField.textProperty().addListener((observable, oldValue, newValue) -> {
            isContentEdited = true;
            isMainTextFieldValidated=false;
        });

        mainFileTextField.focusedProperty().addListener((observable, oldValue, newValue) -> {

            if (!newValue) {
                validateMainTextField();


            }

        });

    }


    private void initAnswersFileTextField() {

        answersFileTextField.textProperty().addListener((observable, oldValue, newValue) -> {
            isContentEdited = true;
        });

        answersFileTextField.focusedProperty().addListener((observable, oldValue, newValue) -> {

            if (!newValue) {
                validateAnswersTextField();
            }

        });
    }

    private void initIdentifierCombo() {
        identifierCombo.setDisable(true);
        identifierCombo.setItems(combosItems);
        identifierCombo.setVisibleRowCount(3);
        identifierCombo.setOnShown(t -> identifierCombo.getSelectionModel().clearSelection());
        identifierCombo.setOnHidden(t -> {
            identifierCombo.getSelectionModel().select(identifierComboSelectedIndex);
            System.out.println("easy" + identifierComboSelectedIndex);
        });
        identifierCombo.getSelectionModel().selectedIndexProperty().addListener((observable, oldValue, newValue) -> {

            if ((Integer) newValue != -1)
                identifierComboSelectedIndex = (Integer) newValue;

        });

        identifierCombo.setCellFactory(
                new Callback<ListView<String>, ListCell<String>>() {
                    @Override
                    public ListCell<String> call(ListView<String> param) {
                        final ListCell<String> cell = new ListCell<String>() {
                            {
                                this.setPrefHeight(rootHeight * 0.06);
                                //this.setFont(new Font("Arial",resY/30));
                            }

                            @Override
                            public void updateItem(String item,
                                                   boolean empty) {
                                super.updateItem(item, empty);
                                if (item != null) {
                                    setText(item);
                                } else {
                                    setText(null);
                                }
                            }
                        };
                        return cell;
                    }

                });
    }


    private void initFormCombo() {
        formCombo.setDisable(true);
        formCombo.setItems(combosItems);
        formCombo.setVisibleRowCount(3);
        formCombo.setOnShown(t -> formCombo.getSelectionModel().clearSelection());
        formCombo.setOnHidden(t -> {
            formCombo.getSelectionModel().select(formComboSelectedIndex);
            System.out.println("easy" + formComboSelectedIndex);
        });
        formCombo.getSelectionModel().selectedIndexProperty().addListener((observable, oldValue, newValue) -> {

            if ((Integer) newValue != -1)
                formComboSelectedIndex = (Integer) newValue;

        });


        formCombo.setCellFactory(
                new Callback<ListView<String>, ListCell<String>>() {
                    @Override
                    public ListCell<String> call(ListView<String> param) {
                        final ListCell<String> cell = new ListCell<String>() {
                            {
                                this.setPrefHeight(rootHeight*0.06);
                                //super.setPrefHeight(identifierCombo.getPrefHeight());
                            }

                            @Override
                            public void updateItem(String item,
                                                   boolean empty) {
                                super.updateItem(item, empty);
                                if (item != null) {
                                    setText(item);
                                } else {
                                    setText(null);
                                }
                            }
                        };
                        return cell;
                    }

                });
    }

    private void populateCombos() {
        ArrayList<String> infoHeaders = CSVHandler.getDetectedInfoHeaders();
        manualColsCounter = 0;

        if(!isOpenMode)
            processInfoHeaders(infoHeaders);
        else
            filteredInfoHeaders=(ArrayList<String>)currentOpenedProjectJson.get(SAVED_INFO_HEADERS_JSON_KEY);

        combosItems.clear();
        combosItems.add(NONE_OPTION);
        manualIDIndex = -1;
        manualFormIndex = -1;

        String idName = "";
        String formName = "";


        if (isOpenMode) {

            idName = (String) currentOpenedProjectJson.get(IDENTIFIER_NAME_JSON_KEY);
            formName = (String) currentOpenedProjectJson.get(FORM_COL_NAME_JSON_KEY);


            if (idName.contains(MANUAL_MODE_INDICATOR)) {
                combosItems.add(idName);
                manualIDIndex = manualColsCounter;
                manualColsCounter++;
            }
            if (formName.contains(MANUAL_MODE_INDICATOR)) {
                combosItems.add(formName);
                manualFormIndex = manualColsCounter;
                manualColsCounter++;
            }


        }

        identifierCombo.getSelectionModel().selectFirst();
        formCombo.getSelectionModel().selectFirst();


        for (String header : filteredInfoHeaders) {
            combosItems.add(header);
            if (!isOpenMode) {
                if (header.toLowerCase().trim().contains("id"))
                    identifierCombo.getSelectionModel().select(combosItems.size() - 1);
                else if (header.toLowerCase().trim().contains("form") || header.toLowerCase().trim().contains("model"))
                    formCombo.getSelectionModel().select(combosItems.size() - 1);
            }
        }

        if (isOpenMode) {
            identifierCombo.getSelectionModel().select(idName);
            formCombo.getSelectionModel().select(formName);
        }


    }


    /*
    prepares the content of combo boxes while handling the complexIDs logic
     */
    private void processInfoHeaders(ArrayList<String> infoHeaders) {

        filteredInfoHeaders = new ArrayList<>();
        int expectedIndex = 1, digitBegin = 0, lastNumberedHeader = -2;
        String currentGroup = "";
        Pattern groupsPattern = Pattern.compile(".*\\d+");
        ArrayList<Group> idGroups = new ArrayList<Group>();
        isComplexIDAdded = false;


        //remove any header having an index and starting with keyword "id" from info headers and add it to IdGroups
        for (int i = 0; i < infoHeaders.size(); i++) {
            if (!groupsPattern.matcher(infoHeaders.get(i)).matches()) { //no groups pattern -> normal header
                if (lastNumberedHeader == i - 1) {
                    int groupSize = expectedIndex - 1;
                    processGroup(groupSize, currentGroup, idGroups, i);
                    expectedIndex = 1;
                    currentGroup = "";
                }
                filteredInfoHeaders.add(infoHeaders.get(i));

            } else { //groups pattern
                lastNumberedHeader = i;
                if ((digitBegin = infoHeaders.get(i).lastIndexOf(Integer.toString(expectedIndex))) == -1) { //expected not found or end of array-> end of group

                    int groupSize = expectedIndex - 1;
                    processGroup(groupSize, currentGroup, idGroups, i);
                    expectedIndex = 2;
                    currentGroup = "";

                } else { //still inside same group
                    currentGroup = infoHeaders.get(i).substring(0, digitBegin);
                    expectedIndex++;
                }
            }
        }

        if (currentGroup.length() != 0) //process last group if exists
            processGroup(expectedIndex - 1, currentGroup, idGroups, lastNumberedHeader + 1);


        //add idGroups to detected groups

        if (idGroups.isEmpty())
            CSVHandler.setRealIDGroups(null);
        else
            CSVHandler.setRealIDGroups(idGroups);

    }


    private void processGroup(int groupSize, String currentGroup, ArrayList<Group> idGroups, int idEnd) {
        if (groupSize > 3 || isComplexIDAdded) { //if a complexID occurred before or consider this a questions group
            if (idGroups.size() == 0) //start of questions groups
                CSVHandler.setQuestionsColStartIndex(idEnd - groupSize);
            idGroups.add(new Group(currentGroup, groupSize,groupSize));
        } else {
            isComplexIDAdded = true;
            complexIdStartIndex = idEnd - groupSize;
            complexIDSize = groupSize;
            filteredInfoHeaders.add(constructComplexIDString(currentGroup, groupSize));
        }
    }

    private void initManualModeToggle() {
        manualModeToggle.getStyleClass().add("blueJFXToggle");
        manualModeToggle.selectedProperty().addListener((observable, oldValue, newValue) ->
        {
            if (newValue) {
                identifierCombo.setDisable(true);
                formCombo.setDisable(true);
            } else {
                if (mainTextFieldResult == CSVFileValidator.SUCCESS) {
                    identifierCombo.setDisable(false);
                    formCombo.setDisable(false);
                } else if (mainTextFieldResult == CSVFileValidator.WARNING) {
                    manualModeToggle.setSelected(true);
                }
            }

        });

    }

    private void saveIdentifierColumn() {


        int identifierSelectedIndex = identifierComboSelectedIndex - 1; //remove None effect

        if (isOpenMode && manualColsCounter > 0 && identifierSelectedIndex != NOT_AVAILABLE) {
            String idName=(String)identifierCombo.getSelectionModel().getSelectedItem();
            if (identifierSelectedIndex == manualIDIndex) { //the id col set chosen in manual mode was selected
                CSVHandler.setIdentifierColStartIndex(Integer.parseInt((String) currentOpenedProjectJson.get(ID_COL_START_INDEX_JSON_KEY)));
                CSVHandler.setIdentifierColEndIndex(Integer.parseInt((String) currentOpenedProjectJson.get(ID_COL_END_INDEX_JSON_KEY)));
                Statistics.setIdentifierName(idName.replace(MANUAL_MODE_INDICATOR,""));
                return;
            } else if (identifierSelectedIndex == manualFormIndex) { //the form col set chosen in manual mode was selected
                int index;
                CSVHandler.setIdentifierColStartIndex(index = Integer.parseInt((String) currentOpenedProjectJson.get(FORM_COL_INDEX_JSON_KEY)));
                CSVHandler.setIdentifierColEndIndex(index + 1);
                Statistics.setIdentifierName(idName.replace(MANUAL_MODE_INDICATOR,""));
                return;
            } else //remove the extra manual cols effect
                identifierSelectedIndex -= manualColsCounter;

        }

        String idenfierName = identifierSelectedIndex == -1 ? "ID" : filteredInfoHeaders.get(identifierSelectedIndex);
        Statistics.setIdentifierName(idenfierName.replace(MANUAL_MODE_INDICATOR, ""));

        if (identifierSelectedIndex == -1) { //none chosen
            CSVHandler.setIdentifierColStartIndex(NOT_AVAILABLE);
            CSVHandler.setIdentifierColEndIndex(NOT_AVAILABLE);
            return;
        }


        int idStartIndex = getUnFilteredIndex(identifierSelectedIndex);
        CSVHandler.setIdentifierColStartIndex(idStartIndex);


        int idEndIndex;

        if (identifierSelectedIndex == complexIdStartIndex && isComplexIDAdded)
            idEndIndex = complexIDSize + idStartIndex;
        else
            idEndIndex = idStartIndex + 1;

        CSVHandler.setIdentifierColEndIndex(idEndIndex);

    }

    private void saveFormColumn() {

        int formSelectedIndex = CSVHandler.getFormsCount() == 1 ? NOT_AVAILABLE : formComboSelectedIndex - 1; //-1 ->remove None effect
        if (formSelectedIndex != NOT_AVAILABLE && isOpenMode && manualColsCounter > 0) {
            int idStart = Integer.parseInt((String)currentOpenedProjectJson.get(ID_COL_START_INDEX_JSON_KEY));
            int idEnd = Integer.parseInt((String)currentOpenedProjectJson.get(ID_COL_END_INDEX_JSON_KEY));

            if (formSelectedIndex == manualFormIndex || (idEnd - idStart > 1 && formSelectedIndex == manualIDIndex)) {
                CSVHandler.setFormColIndex(Integer.parseInt((String)currentOpenedProjectJson.get(FORM_COL_INDEX_JSON_KEY)));
                return;
            } else if (formSelectedIndex == manualIDIndex) {
                CSVHandler.setFormColIndex(Integer.parseInt((String) currentOpenedProjectJson.get(ID_COL_START_INDEX_JSON_KEY)));
                return;
            } else //remove the extra manual cols effect
                formSelectedIndex -= manualColsCounter;
        }
        CSVHandler.setFormColIndex(getUnFilteredIndex(formSelectedIndex));
    }


    private String constructComplexIDString(String groupName, int groupSize) {
        String complexIDString = groupName + " ";
        for (int i = 1; i < groupSize; i++) {
            complexIDString += Integer.toString(i) + "-";
        }
        complexIDString += Integer.toString(groupSize);

        return complexIDString;

    }

    private int getUnFilteredIndex(int filteredIndex) {

        if (!isComplexIDAdded || filteredIndex <= complexIdStartIndex)  //still one to one mapping between filtered and unfiltered info headers
            return filteredIndex;

        return filteredIndex + (complexIDSize - 1);


    }

    private int showHeadersWarningDialog(String fileName) {

        Alert alert = new Alert(Alert.AlertType.WARNING);

        alert.setTitle("No Headers Detected");
        alert.setHeaderText(null);
        alert.setContentText(constructMessage("No headers were detected in ",fileName+" file",". Was the detection correct?"));
        //alert.setOnCloseRequest(t->alert.hide());
        ButtonType yesButton= new ButtonType("Yes, the detection was correct");
        ButtonType  skipRowButton= new ButtonType("No, the file contains headers");
        ButtonType close = ButtonType.CLOSE;


        alert.setGraphic(new ImageView("Images/Error_48px.png"));


        alert.getButtonTypes().setAll(yesButton,skipRowButton, close);

        Button closeButt=((Button)alert.getDialogPane().lookupButton(ButtonType.CLOSE));

        closeButt.setVisible(false);
        closeButt.setMaxWidth(0);
        closeButt.setPrefWidth(0);
        processDialog(alert);
        Optional<ButtonType> result = alert.showAndWait();

        int selected;
        if (result.get() == skipRowButton)
            selected = SKIPROW;
        else if (result.get() == yesButton)
            selected = CONTINUE;
        else
            selected = CANCEL;

        return selected;


    }


    private void validateMainTextField() {

        isMainTextFieldValidated=true;

        CSVFileValidator validator = new CSVFileValidator(mainFileTextField, CSVFileValidator.MAINFILETEXTFIELD);
        mainFileTextField.getValidators().clear();
        mainFileTextField.getValidators().add(validator);

        mainFileTextField.validate();

        mainTextFieldResult = validator.getMessageType();
        mainTextFieldMessage = validator.getMessage();

        if (validator.getMessageType() == ValidatorBase.SUCCESS) {

            populateCombos();
            manualModeToggle.setSelected(false);
            formCombo.setDisable(false);
            identifierCombo.setDisable(false);
            generateAnswerKeyCSVFromFirstLine(true);
        } else {
            if (validator.getMessageType() == ValidatorBase.WARNING){
                manualModeToggle.setSelected(true);
                generateAnswerKeyCSVFromFirstLine(false);
            }

            formCombo.setDisable(true);
            identifierCombo.setDisable(true);
        }
    }

    private void validateAnswersTextField() {
        if(isQuestMode || isAnswerKeyInFirstRow)
            return;

        CSVFileValidator validator = new CSVFileValidator(answersFileTextField, CSVFileValidator.ANSWERSFILETEXTFIELD);
        answersFileTextField.getValidators().clear();
        answersFileTextField.getValidators().add(validator);
        answersFileTextField.validate();
        answersTextFieldResult = validator.getMessageType();
        answersTextFieldMessage = validator.getMessage();
    }

    private void openManualMode() {
        CSVHandler.setRealIDGroups(null);//populate combos wasn't called -> reintialize realIDGroups
        CSVHandler.setIdentifierColStartIndex(NOT_AVAILABLE);
        CSVHandler.setIdentifierColEndIndex(NOT_AVAILABLE);
        CSVHandler.setFormColIndex(NOT_AVAILABLE);
        new ManualModeController(this).startWindow();

    }


}
