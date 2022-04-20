package com.kjain.flandroidclient

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        val tempStr = get_text()

        val helloTextView = findViewById(R.id.hellotext) as TextView;
        helloTextView.setText(tempStr)
    }

    private fun get_text(): String {
        val py = Python.getInstance()
        val module = py.getModule("client")
        return module.callAttr("client_train").toString()
    }
}
