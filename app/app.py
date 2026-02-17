import os

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import SubmitField

from app.utils.cleanup import clear_uploads, register_cleanup
from app.utils.predict import predict

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = "wnrhbnwrzngbwzgv"
app.config["UPLOADED_PHOTOS_DEST"] = os.path.join(BASE_DIR, "uploads")

clear_uploads()
register_cleanup()

photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, "Dopuštene su samo slike"),
            FileRequired("Polje datoteke ne smije biti prazno"),
        ]
    )
    submit = SubmitField("Upload")


@app.route("/uploads/<filename>")
def get_file(filename: str):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)


@app.route("/", methods=["GET", "POST"])
def index():
    form: UploadForm = UploadForm()
    prediction: str | None = None
    file_url: str | None = None

    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        return redirect(url_for("index", filename=filename))

    filename = request.args.get("filename")
    if filename:
        file_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], filename)
        file_url = url_for("get_file", filename=filename) if filename else None
        prediction = predict(file_path)

    return render_template(
        "index.html", form=form, file_url=file_url, prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
