import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {catchError, map} from "rxjs/operators";


@Injectable({
  providedIn: 'root'
})
export class ImageUploadService {
  url = "https://kemperino.com/api/upload"
  constructor(private http:HttpClient) { }

  uploadImage(img) {
    var url = this.url.concat('/images/')
    return this.http.post(url, img)
      .pipe(map(res => res));

  }
  getImage(id){
       var url = this.url.concat('/get/' + id);

       return this.http.get(url)
      .pipe(map(res => res));
  }
}
