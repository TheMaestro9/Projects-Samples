package Jowil;

import com.jfoenix.controls.JFXListCell;
import de.jensd.fx.glyphs.GlyphsBuilder;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIcon;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIconView;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.TableView;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.text.Font;


import static Jowil.Controller.resX;

class OpenProjectCell extends JFXListCell {


    private long lastClick;
    String myText;
    Dialog dialog;
    private final  Node removeIcon=GlyphsBuilder.create(FontAwesomeIconView.class).glyph(FontAwesomeIcon.TRASH_ALT).size(Double.toString(resX/80)).styleClass("projectRemoveIcon").build();
    private final  Node openFolderIcon=GlyphsBuilder.create(FontAwesomeIconView.class).glyph(FontAwesomeIcon.EXTERNAL_LINK).size(Double.toString(resX/80)).styleClass("projectOpenIcon").build();


    StartController parentController;

    OpenProjectCell(Dialog dialog,StartController parentController) {
        super();
        this.dialog = dialog;
        this.parentController=parentController;



    }
    @Override
    public void updateItem(Object item,boolean empty){

//        if(item!=null ) {
//            if(((String)item).isEmpty())
//                this.setGraphic(null);
//            else {
//                label.setText((String) item);
//                myText = (String) item;
//            }
//        }

        super.updateItem("",empty);
        if(item!=null){
            AnchorPane cellPane=new AnchorPane();
            cellPane.setStyle("-fx-background-color:transparent;");
            HBox leftHbox = new HBox(5);
            HBox rightHBox=new HBox(11);

            this.setOnMouseClicked(event -> {

                if (Math.abs(lastClick - (lastClick = System.currentTimeMillis())) < 700)
                    parentController.startOpenMode(myText,dialog);

            });

            ImageView imageView = new ImageView();
            imageView.setImage(new Image("Images/Folder_96px.png"));
            imageView.setFitWidth(resX*20/1280);
            imageView.setFitHeight(resX*20/1280);

            Label label = new Label(myText=(String)item);
            label.setId("label");
            label.setFont(new Font(resX*15/1280));
            label.setAlignment(Pos.CENTER);

            StackPane deleteButton=new StackPane();
            deleteButton.getChildren().add(removeIcon);
            TranslatableTooltip tooltipRemove = new TranslatableTooltip("Delete Project");

            Tooltip.install(deleteButton, tooltipRemove);
            deleteButton.setOnMouseClicked(event ->{
                lastClick=0;
                parentController.deleteProject(myText);});

            StackPane openButton=new StackPane();
            openButton.getChildren().add(openFolderIcon);
            TranslatableTooltip tooltipOpen = new TranslatableTooltip("Show Project in Explorer");

            Tooltip.install(openButton, tooltipOpen);
            openButton.setOnMouseClicked(event ->{
                lastClick=0;
                parentController.openProjectInExplorer(myText);
            });

            leftHbox.getChildren().addAll(imageView, label);
            rightHBox.getChildren().addAll(openButton,deleteButton);
            cellPane.getChildren().addAll(leftHbox,rightHBox);
            AnchorPane.setLeftAnchor(leftHbox,0.0);
            AnchorPane.setRightAnchor(rightHBox,2.0);

            this.setGraphic(cellPane);
        }


    }

    public static OpenProjectCell createOpenProjectCell(Dialog dialog,StartController parentController) {
        return new OpenProjectCell(dialog,parentController);
    }
}
